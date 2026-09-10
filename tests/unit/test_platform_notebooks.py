"""Static-only gate for notebooks/platform/*.ipynb (D-17 — no cell execution).

No `nbmake`, no `papermill`, no notebook-execution API of any kind. Every
test is parameterized over ``sorted(Path("notebooks/platform").glob("*.ipynb"))``
so a notebook added by a later plan is covered automatically without
editing this file — the only exception is the A13-discipline guard, which
names three specific future notebooks by design (it tightens automatically
as they land, guarded by an existence check).

Checks:
    1. Every notebook parses as valid nbformat and has at least one cell.
    2. Criterion 3: no code cell imports matplotlib/seaborn directly or uses
       bare ``plt.``/``sns.`` attribute access — all figure construction
       routes through ``platform/plotting/``.
    3. D-10: no code cell calls ``.save(`` — a notebook may never write a
       production checkpoint.
    4. T-06-01 (secret hygiene): a notebook that loads the platform config
       must also reference ``redacted_config`` somewhere, and no cell may
       display the whole config object via ``print(cfg)`` or a trailing
       bare ``cfg`` expression.
    5. D-10 header convention: every notebook's first cell is markdown and
       names a prerequisite command under ``scripts/``.
    6. Cross-notebook A13 discipline: P3/P4/P6 must mention audit item A13
       wherever the §5.4 lag/ratio headline appears, once each notebook
       exists.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import nbformat
import pytest

NOTEBOOK_DIR = Path("notebooks/platform")
NOTEBOOK_PATHS = sorted(NOTEBOOK_DIR.glob("*.ipynb")) if NOTEBOOK_DIR.exists() else []

# Criterion 3 (D-01 / ADR #11 applied to the platform): no notebook cell may
# construct a figure directly — all plotting routes through platform/plotting/.
_FORBIDDEN_PLOTTING_TOKENS = (
    "import matplotlib",
    "import seaborn",
    "plt.",
    "sns.",
)

_CHECKPOINT_WRITE_TOKEN = ".save("

# Deliberately hard-coded — the one sanctioned exception to "no literal
# notebook filename outside this list" (see module docstring). These three
# notebooks do not exist yet in this plan's wave; the existence guard below
# lets this test pass now and tighten automatically once they land.
_A13_GATED_NOTEBOOKS = (
    "P3_regime_labeling.ipynb",
    "P4_nowcaster.ipynb",
    "P6_backtest_evaluation.ipynb",
)


def _load(nb_path: Path) -> nbformat.NotebookNode:
    return nbformat.read(nb_path, as_version=4)


def _code_cells(nb: nbformat.NotebookNode) -> list:
    return [cell for cell in nb.cells if cell.cell_type == "code"]


def _has_bare_cfg_expression(source: str) -> bool:
    """True if *source*'s top level ends with a bare `cfg` expression (Jupyter would display it)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Name) and node.value.id == "cfg":
            return True
    return False


# ── (1) Parses as valid nbformat ─────────────────────────────────────────────

@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_parses_and_has_cells(nb_path: Path):
    nb = _load(nb_path)
    assert len(nb.cells) >= 1, f"{nb_path.name}: notebook has no cells"


# ── (2) Criterion 3: no direct plotting-library use in a notebook cell ──────

@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_has_no_forbidden_plotting_tokens(nb_path: Path):
    nb = _load(nb_path)
    offenses = []
    for idx, cell in enumerate(_code_cells(nb)):
        for token in _FORBIDDEN_PLOTTING_TOKENS:
            if token in cell.source:
                offenses.append((idx, token))
    assert not offenses, (
        f"{nb_path.name}: forbidden plotting token(s) found in cell(s) — all figure "
        f"construction must route through platform/plotting/: {offenses}"
    )


# ── (3) D-10: never write a production checkpoint ───────────────────────────

@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_never_writes_a_production_checkpoint(nb_path: Path):
    nb = _load(nb_path)
    offenses = [
        idx for idx, cell in enumerate(_code_cells(nb)) if _CHECKPOINT_WRITE_TOKEN in cell.source
    ]
    assert not offenses, (
        f"{nb_path.name}: cell(s) {offenses} call '.save(' — notebooks may never write a "
        f"production checkpoint (D-10)."
    )


# ── (4) T-06-01: secret hygiene ──────────────────────────────────────────────

@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_never_displays_unredacted_config(nb_path: Path):
    nb = _load(nb_path)
    code_cells = _code_cells(nb)
    loads_config = any("load_platform_config" in cell.source for cell in code_cells)
    if not loads_config:
        return

    uses_redacted = any("redacted_config" in cell.source for cell in code_cells)
    assert uses_redacted, (
        f"{nb_path.name}: calls load_platform_config() but never references "
        f"redacted_config() anywhere in the notebook (T-06-01)."
    )

    for idx, cell in enumerate(code_cells):
        assert "print(cfg)" not in cell.source, (
            f"{nb_path.name}: cell {idx} contains 'print(cfg)' — this displays a whole "
            f"config object that may carry a live FRED_API_KEY (T-06-01)."
        )
        assert not _has_bare_cfg_expression(cell.source), (
            f"{nb_path.name}: cell {idx} ends with a bare 'cfg' expression, which Jupyter "
            f"displays via repr() and may carry a live FRED_API_KEY (T-06-01)."
        )


# ── (5) D-10 header convention ───────────────────────────────────────────────

@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_first_cell_is_markdown_with_prerequisite(nb_path: Path):
    nb = _load(nb_path)
    first_cell = nb.cells[0]
    assert first_cell.cell_type == "markdown", f"{nb_path.name}: first cell must be markdown"
    assert "scripts/" in first_cell.source, (
        f"{nb_path.name}: first cell must name a prerequisite build command under scripts/ (D-10)."
    )


# ── (6) Cross-notebook A13 discipline ────────────────────────────────────────

@pytest.mark.parametrize("nb_name", _A13_GATED_NOTEBOOKS)
def test_a13_discipline_notebooks_mention_audit_item(nb_name: str):
    nb_path = NOTEBOOK_DIR / nb_name
    if not nb_path.exists():
        pytest.skip(f"{nb_name} not yet built (a later plan produces it)")
    nb = _load(nb_path)
    combined_source = "\n".join(cell.source for cell in nb.cells)
    assert "A13" in combined_source, (
        f"{nb_name}: must mention audit item A13 wherever the §5.4 lag/ratio headline appears."
    )


# ── D-17: this module tests notebooks WITHOUT executing them ───────────────

def test_this_module_uses_no_execution_based_testing():
    assert "nbmake" not in sys.modules
    assert "papermill" not in sys.modules
