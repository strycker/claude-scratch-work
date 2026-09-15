"""Ratchet guard on ``platform/``'s remaining legacy-library imports (criterion 8).

**Why this file exists.** `MIGRATION-PLAN.md`:98 defined the decoupling exit criterion as::

    grep -r "from trading_crab_lib\\." src/trading_crab_lib/platform | grep -v platform

returning nothing. That check **can never fail**: every match line begins with the path
``src/trading_crab_lib/platform/...``, which itself contains the substring ``platform``, so
``grep -v platform`` discards every line — genuine violations included. It returned nothing
on 2026-09-10 and it returns nothing today, with 31 real legacy imports in the tree. That
false negative is what backed ROADMAP criterion 8's "Verified 2026-09-10: platform is
currently fully decoupled", and it is exactly the evidence shape ``UAT-AUDIT-2026-09-09``
warns about: a check that confirms rather than tests.

**What this asserts instead.** An AST scan (not a substring grep) counts real import
statements. The count is pinned as a **ratchet**: it may fall, never rise. This deliberately
does NOT assert zero — the 31 sites are pre-existing, predate Phase 7, and vendoring them is
`MIGRATION-PLAN.md` P0 / Phase 8 criterion 1. Asserting zero today would just be a red test
carrying no new information. Asserting "no worse than today" catches the regression that
matters now: a new phase quietly adding the 32nd.

Lower ``MAX_LEGACY_IMPORT_SITES`` as seams are vendored. Never raise it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[2] / "src" / "trading_crab_lib" / "platform"

#: Measured 2026-09-15 by the scan below. RATCHET — may only decrease.
MAX_LEGACY_IMPORT_SITES = 31

#: The four seams MIGRATION-PLAN.md P0 names for vendoring.
EXPECTED_SEAMS = {
    "trading_crab_lib",
    "trading_crab_lib.checkpoints",
    "trading_crab_lib.ingestion",
    "trading_crab_lib.ingestion.assets",
    "trading_crab_lib.ingestion.browser",
    "trading_crab_lib.ingestion.http",
    "trading_crab_lib.email",
}


def _legacy_import_sites() -> list[tuple[str, int, str]]:
    """Every `import`/`from` in platform/ naming a NON-platform trading_crab_lib module."""
    sites: list[tuple[str, int, str]] = []
    for path in sorted(PLATFORM_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else []
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module.startswith("trading_crab_lib") and not module.startswith("trading_crab_lib.platform"):
                    sites.append((str(path.relative_to(PLATFORM_ROOT)), node.lineno, module))
    return sites


class TestLegacyImportRatchet:
    def test_platform_root_exists(self):
        """Guard the guard: a wrong path would make every scan below vacuously pass."""
        assert PLATFORM_ROOT.is_dir(), f"platform root not found at {PLATFORM_ROOT}"
        assert list(PLATFORM_ROOT.rglob("*.py")), "no .py files scanned — the glob is broken"

    def test_legacy_import_count_does_not_grow(self):
        """The rejected value is 32 — one new legacy import added by a later phase."""
        sites = _legacy_import_sites()
        assert len(sites) <= MAX_LEGACY_IMPORT_SITES, (
            f"platform/ legacy imports rose to {len(sites)} (ratchet: {MAX_LEGACY_IMPORT_SITES}). "
            f"platform/ must not import from the legacy library (ROADMAP criterion 8). "
            f"New sites:\n"
            + "\n".join(f"  {f}:{ln} -> {m}" for f, ln, m in sites[MAX_LEGACY_IMPORT_SITES:])
        )

    def test_no_unexpected_seam_appears(self):
        """A brand-new legacy subpackage is a bigger regression than one more call site."""
        found = {module for _f, _ln, module in _legacy_import_sites()}
        unexpected = found - EXPECTED_SEAMS
        assert not unexpected, (
            f"platform/ imports legacy modules outside MIGRATION-PLAN.md's known seams: "
            f"{sorted(unexpected)}. Vendor it rather than widening the coupling."
        )

    def test_the_broken_grep_is_not_the_exit_criterion(self):
        """MIGRATION-PLAN.md must not present the unfalsifiable grep as its exit check.

        Tests the **exit criterion line specifically**, not the whole document: the file
        legitimately quotes the broken command inside a warning explaining why it is broken,
        and a naive substring search over the whole text would flag that warning as the
        defect it documents. What must not recur is the grep being relied upon.
        """
        plan_path = PLATFORM_ROOT.parents[2] / "MIGRATION-PLAN.md"
        exit_lines = [
            line
            for line in plan_path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("**Exit:**")
        ]
        assert exit_lines, "no **Exit:** criterion found in MIGRATION-PLAN.md"
        offenders = [line for line in exit_lines if "grep -v platform" in line]
        assert not offenders, (
            "MIGRATION-PLAN.md relies on `| grep -v platform` as a decoupling exit check:\n"
            + "\n".join(f"  {line.strip()}" for line in offenders)
            + "\nThat filter discards every match line (each begins with a path containing "
            "'platform'), so the check cannot fail. Use the AST scan in this module instead."
        )


@pytest.mark.parametrize("module", sorted(EXPECTED_SEAMS))
def test_each_expected_seam_is_real(module):
    """Every seam in EXPECTED_SEAMS is actually imported — keeps the allowlist honest.

    Without this, a vendored seam would linger in EXPECTED_SEAMS forever and silently
    re-permit a coupling that had already been removed.
    """
    found = {m for _f, _ln, m in _legacy_import_sites()}
    assert module in found, (
        f"'{module}' is in EXPECTED_SEAMS but nothing in platform/ imports it any more. "
        f"It was vendored — remove it from the allowlist and lower MAX_LEGACY_IMPORT_SITES."
    )
