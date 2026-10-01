"""Boundary tests for ``platform/parked/`` (08.2-02, D-06; threat T-08.2-05).

Parked research code (classifier #2, the joint driver, the stability suite) must be
unreachable from the weekly product. Three independent checks:

- **Runtime:** a fresh interpreter imports every module under ``platform/report/`` and
  ``platform/tripwire/`` (glob-discovered, so a new module is covered with no edit) and
  dumps ``sys.modules``. Nothing from ``platform.parked``, ``allocation.joint_tilt``,
  ``evaluation.churn`` or ``evaluation.dependence`` may be loaded.
- **Static:** an AST scan of every file under ``src/trading_crab_lib/platform/`` outside
  ``parked/`` for an import of ``platform.parked`` at any depth. This catches the lazy,
  function-level import that the runtime check cannot see. The scanner has a self-check.
- **Location:** each parked module is found under ``parked`` and gone from its old path.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLATFORM_DIR = REPO / "src" / "trading_crab_lib" / "platform"
PKG = "trading_crab_lib.platform"
PARKED = f"{PKG}.parked"

FORBIDDEN_EXACT = {
    f"{PKG}.allocation.joint_tilt",
    f"{PKG}.evaluation.churn",
    f"{PKG}.evaluation.dependence",
}

# module -> its pre-08.2-02 path (must no longer import)
PARKED_MODULES = {
    "stability": f"{PKG}.labeling.stability",
    "joint_driver": f"{PKG}.backtest.joint_driver",
}


def _served_modules() -> list[str]:
    """Dotted names of every module under report/ and tripwire/, discovered by glob."""
    names: list[str] = []
    for sub in ("report", "tripwire"):
        for path in sorted((PLATFORM_DIR / sub).glob("*.py")):
            stem = path.stem
            names.append(f"{PKG}.{sub}" if stem == "__init__" else f"{PKG}.{sub}.{stem}")
    return names


def _parked_import_lines(source: str) -> list[int]:
    """Line numbers of every import (any depth) that names ``platform.parked``."""
    hits: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(a.name == PARKED or a.name.startswith(PARKED + ".") for a in node.names):
                hits.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and (mod == PARKED or mod.startswith(PARKED + ".")):
                hits.append(node.lineno)
            elif node.level == 0 and mod == PKG and any(a.name == "parked" for a in node.names):
                hits.append(node.lineno)
    return hits


# ── Runtime: the served import graph never loads parked code ────────────────────────
def test_served_modules_are_discovered():
    names = _served_modules()
    assert f"{PKG}.report.weekly" in names
    assert f"{PKG}.tripwire.monitor" in names


def test_served_import_graph_loads_nothing_parked():
    modules = _served_modules()
    code = (
        "import importlib, json, sys\n"
        f"for m in {modules!r}:\n"
        "    importlib.import_module(m)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    loaded = set(json.loads(proc.stdout.strip().splitlines()[-1]))

    # positive control: the subprocess really imported the served graph
    assert f"{PKG}.report.weekly" in loaded
    bad = sorted(k for k in loaded if k.startswith(PARKED) or k in FORBIDDEN_EXACT)
    assert not bad, f"served graph reaches parked or parked-only code: {bad}"


# ── Static: no import of platform.parked outside parked/ ────────────────────────────
def test_scanner_flags_a_function_level_parked_import():
    src = (
        "def lazy():\n"
        "    from trading_crab_lib.platform.parked.joint_driver import run_joint_backtest\n"
        "    return run_joint_backtest\n"
    )
    assert _parked_import_lines(src) == [2]
    assert _parked_import_lines("import trading_crab_lib.platform.parked.stability\n") == [1]
    assert _parked_import_lines("from trading_crab_lib.platform import parked\n") == [1]
    # and it does not over-flag an ordinary import
    assert _parked_import_lines("from trading_crab_lib.platform.labeling import jump\n") == []


def test_no_src_module_outside_parked_imports_parked():
    parked_dir = PLATFORM_DIR / "parked"
    files = [p for p in PLATFORM_DIR.rglob("*.py") if parked_dir not in p.parents]
    assert len(files) >= 50, f"scanner only saw {len(files)} files; glob is wrong"
    offenders = {
        str(p.relative_to(REPO)): lines
        for p in files
        if (lines := _parked_import_lines(p.read_text(encoding="utf-8")))
    }
    assert not offenders, f"active src imports parked code: {offenders}"


# ── Location: moved, not copied ─────────────────────────────────────────────────────
@pytest.mark.parametrize("name, old_path", sorted(PARKED_MODULES.items()))
def test_module_lives_only_under_parked(name, old_path):
    assert importlib.util.find_spec(f"{PARKED}.{name}") is not None
    assert importlib.util.find_spec(old_path) is None


def test_quality_tier_moved_not_copied():
    """The A11 gate lives in evaluation/deflated_sharpe.py; parked code re-imports the same objects."""
    from trading_crab_lib.platform.evaluation import deflated_sharpe
    from trading_crab_lib.platform.parked import joint_driver

    for name in ("quality_tier", "annualized_sharpe", "QUALITY_TIER_RULE"):
        assert getattr(joint_driver, name) is getattr(deflated_sharpe, name), name
