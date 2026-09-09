# Phase 6: Platform Notebook Suite - Pattern Map

**Mapped:** 2026-09-09
**Files analyzed:** ~15 new files (plotting package: 9 files, notebooks: 6, tests: 3) + 1 modified
**Analogs found:** 15 / 15

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `platform/plotting/__init__.py` | utility (re-export barrel) | transform | `trading_crab_lib/plotting/__init__.py` | exact (shape), diverges on signature |
| `platform/plotting/core.py` | utility (save/show/palette) | transform | `trading_crab_lib/plotting/core.py` | exact (shape), diverges on signature (no RunConfig) |
| `platform/plotting/data.py` (P1) | component (plot fns) | transform | `trading_crab_lib/plotting/ingestion.py` | role-match |
| `platform/plotting/features.py` (P2) | component (plot fns) | transform | `trading_crab_lib/plotting/features.py` | role-match |
| `platform/plotting/regime.py` (P3, incl. A13 side-by-side) | component (plot fns) | transform | `trading_crab_lib/plotting/regime.py` | role-match |
| `platform/plotting/nowcaster.py` (P4) | component (plot fns) | transform | `trading_crab_lib/plotting/prediction.py` | role-match |
| `platform/plotting/allocation.py` (P5) | component (plot fns) | transform | `trading_crab_lib/plotting/assets.py` | role-match |
| `platform/plotting/backtest.py` (P6) | component (plot fns) | transform | `trading_crab_lib/plotting/diagnostics.py` (RRG/dashboards) + legacy `prediction.py` (KPI-style bars) | role-match |
| `platform/plotting/drift.py` (D-09 drift + D-11 plausibility) | utility (pure functions) | transform | `platform/splice.py::assert_yield_units_plausible` + `tests/integration/test_mini_backtest.py::TestKpisAreOnAPlausibleScale` | role-match (new logic class, no direct plotting analog) |
| `notebooks/platform/P1_data_spine.ipynb` | notebook (app) | request-response (interactive) | `notebooks/01_ingestion.ipynb` | exact (structure), diverges on data source (full-span opt-in) |
| `notebooks/platform/P2_features_taxonomy.ipynb` | notebook (app) | request-response | `notebooks/02_features.ipynb` | exact (structure) |
| `notebooks/platform/P3_regime_labeling.ipynb` | notebook (app) | request-response | `notebooks/04_regimes.ipynb` | role-match (regime timeline/transitions), + new sign-off cell (no analog) |
| `notebooks/platform/P4_nowcaster.ipynb` | notebook (app) | request-response | `notebooks/05_prediction.ipynb` | role-match |
| `notebooks/platform/P5_assets_allocation.ipynb` | notebook (app) | request-response | `notebooks/06_assets.ipynb` | role-match |
| `notebooks/platform/P6_backtest_evaluation.ipynb` | notebook (app) | request-response | `notebooks/09_diagnostics.ipynb` / `notebooks/10_model_comparison.ipynb` | role-match |
| `tests/unit/test_platform_plotting.py` | test | transform | `tests/unit/test_plotting.py` | exact |
| `tests/unit/test_platform_plotting_drift.py` | test | transform | `tests/unit/test_platform_splice.py` (`assert_yield_units_plausible` tests) + `tests/integration/test_mini_backtest.py::TestKpisAreOnAPlausibleScale` | role-match |
| `tests/unit/test_platform_notebooks.py` | test (static, no execution) | transform | none direct — new shape; closest precedent is `tests/test_pipeline_smoke.py`'s "parses/imports resolve without running" idiom | partial |
| `platform/evaluation/report.py` (MODIFY — additive only) | service (pipeline-shaped, persists artifacts) | batch | `write_backtest_report()` / `run_full_backtest_evaluation()` in the same file | exact — this is a self-modification, not cross-file |

## Pattern Assignments

### `platform/plotting/core.py` (utility, transform)

**Analog:** `src/trading_crab_lib/plotting/core.py` (pattern source only — D-01 forbids importing it)

**Legacy imports pattern** (lines 19-53):
```python
from __future__ import annotations
import json, logging
from datetime import datetime
from pathlib import Path
from trading_crab_lib import OUTPUT_DIR
from trading_crab_lib import checkpoints as _checkpoints_mod
from trading_crab_lib.runtime import RunConfig

def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

try:
    import matplotlib
    if not _in_jupyter():
        matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
except ImportError as _matplotlib_err:
    raise ImportError(
        "matplotlib is required for plotting functions. "
        "Install with: pip install 'trading-crab-lib[plotting]'"
    ) from _matplotlib_err
```
**Copy verbatim:** the `_in_jupyter()` guard, the `matplotlib.use("Agg")` conditional, and the
`ImportError` install-hint pattern. These have zero platform-specific coupling.

**Legacy palette + `_save_or_show`** (lines 57-83):
```python
CUSTOM_COLORS: list[str] = ["#0000d0", "#d00000", "#f48c06", "#8338ec", "#50a000"]
REGIME_CMAP = mcolors.ListedColormap(CUSTOM_COLORS)
PLOT_DIR = OUTPUT_DIR / "plots"

def _save_or_show(fig: plt.Figure, filename: str, run_cfg: RunConfig) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if run_cfg.save_plots:
        out = PLOT_DIR / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved plot: %s", out)
    if run_cfg.show_plots or _in_jupyter():
        plt.show()
    plt.close(fig)

def _regime_color(cluster_id: int) -> str:
    return CUSTOM_COLORS[cluster_id % len(CUSTOM_COLORS)]
```

**REQUIRED DIVERGENCE — no `RunConfig` on the platform side.**
Verified directly: `platform/config.py` (`load_platform_config()`, lines 56-94) returns a plain
`dict[str, Any]` from `yaml.safe_load()` — no dataclass, no `save_plots`/`show_plots` flags, no
object threaded through the pipeline. `grep -rn "RunConfig" src/trading_crab_lib/platform/`
returns nothing. D-02 (CONTEXT.md) is therefore not a stylistic preference but the *only*
signature that has anything to bind to. The platform equivalent, per D-02 + research Pattern 1:

```python
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def _save_or_show(fig: plt.Figure, *, save_path: Path | None, show: bool) -> plt.Figure:
    """Finalize a figure per D-02: caller decides save/show explicitly, no RunConfig."""
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show or _in_jupyter():
        plt.show()
    return fig  # do NOT plt.close(fig) — the caller/notebook needs the Figure returned (D-02)
```

Every `plot_x` function in every platform submodule must follow:
`plot_x(data, *, save_path: Path | None = None, show: bool = False) -> plt.Figure`.
Do **not** add `run_cfg`, `PLOT_DIR` module constant tied to `OUTPUT_DIR`, or the
`_plot_is_fresh`/`load_or_generate` staleness-caching machinery (legacy lines 88-167) — that
machinery exists to avoid re-running expensive `run_cfg`-driven pipeline steps and has no
platform equivalent to key off; skip it unless a later task explicitly asks for caching.

**Palette:** reuse `CUSTOM_COLORS`/`REGIME_CMAP` verbatim (five colors — matches platform's
K=5 default) unless Claude's Discretion picks a fresh palette; either is acceptable per
CONTEXT.md.

### `platform/plotting/__init__.py` (utility, re-export barrel)

**Analog:** `src/trading_crab_lib/plotting/__init__.py`

**Pattern** (lines 19-171, structure only): group imports by source notebook (`# ── Step 0N ── ` comment banners, en-dash style per CLAUDE.md conventions), import every public plot function
explicitly (no `import *`), and build an explicit `__all__` list mirroring the import groups.
Adapt the banner labels to P1..P6 instead of Step 01-09:

```python
# ── P1: Data Spine ────────────────────────────────────────────────────────────
from trading_crab_lib.platform.plotting.data import (...)
# ── P2: Features & Taxonomy ────────────────────────────────────────────────────
from trading_crab_lib.platform.plotting.features import (...)
# ── Core helpers and constants ─────────────────────────────────────────────────
from trading_crab_lib.platform.plotting.core import (CUSTOM_COLORS, REGIME_CMAP, ...)
```
Do not re-export `PLOT_DIR`/`load_or_generate`/`list_available_plots` — those are `OUTPUT_DIR`-
and-`RunConfig`-coupled legacy concepts with no platform equivalent (see divergence note above).

### `platform/plotting/drift.py` (utility, pure functions — D-09 + D-11)

**Analog 1 — plausibility raise/warn shape:** `src/trading_crab_lib/platform/splice.py:124-155`
(`assert_yield_units_plausible`), quoted in full:
```python
_IMPLAUSIBLE_DECIMAL_YIELD = 1.0
_SUSPICIOUSLY_LOW_MEDIAN_YIELD = 0.001

def assert_yield_units_plausible(yields_decimal: pd.Series, *, source: str) -> None:
    """Fail loudly on an unambiguous yield-units error; warn on a suspicious one."""
    clean = yields_decimal.dropna()
    if clean.empty:
        return
    worst = float(clean.abs().max())
    if worst > _IMPLAUSIBLE_DECIMAL_YIELD:
        raise ValueError(
            f"{source}: yield of {worst:.4g} ({worst:.1%} annualized) after unit conversion is "
            f"not a plausible rate. The series is almost certainly in PERCENT while configured "
            f"as decimal — set `yield_units: percent` for this splice class. Left uncaught this "
            f"compounds into a nonsense total-return index rather than failing."
        )
    median = float(clean.median())
    if median < _SUSPICIOUSLY_LOW_MEDIAN_YIELD:
        log.warning(
            "%s: median yield after conversion is %.6g (%.4f%% annualized), which is very low. ...",
            source, median, median * 100,
        )
```

**Copy this exact shape for every D-11 plausibility function:** module-level named constants
for the bound (not magic numbers inline), a hard `raise ValueError` with an f-string explaining
*why* the value is impossible and what to check, and (only where the failure direction is
ambiguous, per this file's own reasoning) a `log.warning` for the suspicious-but-not-impossible
case. Do not invent a new convention (e.g. returning a bool, or a dataclass verdict) — this
raise/warn asymmetry is the established idiom.

**Values to hard-code as named constants** (from `06-VALIDATION.md` plausibility table — Claude's
Discretion picks exact numbers, but these are the starting proposal, already verified against
current on-disk artifacts):
- `terminal_log_wealth`: universal `abs(x) < 10.0`; domain `[-3, 12]` per leg
- `max_drawdown`: universal `x ∈ [-1, 0]`; domain per-leg floors (buy-and-hold `< -0.30`,
  60/40 `< -0.10`, Faber-style trend-following `> -0.50`)
- `brier_multiclass` (K=5): `x ∈ [0, 1]` — **not [0,2]**; no-skill floor `(K-1)/K² = 0.16`
- `turnover`: `x ∈ [0, 2]`; operational `< 0.30`
- `cvar_5pct`: `x ∈ [-0.5, 0]`; operational `[-0.15, -0.01]`
- `regime_occupancy`: each `∈ [0,1]`, `Σ = 1.0` exactly; soft warn `< 0.05`
  (mirror existing `_MIN_OCCUPANCY_THRESHOLD` in `platform/labeling/diagnostics.py:67`)

**Analog 2 — the multiclass Brier formula that any Brier band must match** (`platform/evaluation/model_metrics.py:58-81`, quoted verbatim per research):
```python
diff = proba - onehot
return float(np.mean(diff * diff))
```
This is `mean` over the full `(n, K)` array — bound is `[0, 1]`, NOT the textbook `[0, 2]`.
A plausibility band written against `[0, 2]` will be silently wrong (Pitfall 3 in RESEARCH.md).

**Regression test requirement (from VALIDATION.md, non-negotiable):** `test_platform_plotting_drift.py`
must include the two named historical failures as explicit in-band/out-of-band pairs:
- a 60/40-shaped leg at `-2.27%` max DD **must FAIL** its domain band (even though it passes the
  universal `∈[-1,0]` bound — this is the whole point of having both a universal and domain band)
- a terminal log wealth of `111.06` **must FAIL** the universal `abs(x)<10.0` band

**D-09 drift statistic:** no existing codebase analog (genuinely new); Claude's Discretion per
CONTEXT.md picks the statistic (standardized mean shift / KS distance / rolling z vs baseline σ).
Whatever is chosen must be a pure function `compute_drift(current: pd.Series, baseline: pd.Series) -> dict`
with no side effects, tested the same does-not-crash + edge-case way as the plotting functions.
**Explicit non-goal:** do not let D-09 alone stand in for D-11 — VALIDATION.md's standing lesson
table shows drift-only would have caught *neither* historical failure (a uniformly-wrong series
shows zero drift against itself).

### `tests/unit/test_platform_plotting.py` (test)

**Analog:** `tests/unit/test_plotting.py` (verbatim pattern, lines 1-80 read in full)

**Imports + Agg-backend pattern** (lines 7-23):
```python
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pytest

from trading_crab_lib import plotting
from trading_crab_lib.plotting import core as _plotting_core
from trading_crab_lib.runtime import RunConfig
```
**Platform divergence:** drop the `RunConfig` import and the `run_cfg` fixture entirely (no
`RunConfig` exists on the platform side). Replace with direct `save_path=tmp_path / "x.png"`,
`show=False` kwargs per test, matching the new `plot_x(..., *, save_path, show)` signature.

**Fixture pattern to copy directly** (lines 36-63) — synthetic frame fixtures with a fixed
`np.random.default_rng(42)` seed, small row counts (40), index via `pd.date_range(..., freq="QE")`
(swap to `"ME"` monthly for platform data, since `monthly_features` is monthly not quarterly):
```python
@pytest.fixture
def raw_df():
    idx = pd.date_range("2000-03-31", periods=40, freq="QE")
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((40, 5)), index=idx, columns=[...])
```

**Does-not-crash + empty-input test class pattern** (lines 66-80, `TestSaveOrShow`):
```python
class TestSaveOrShow:
    def test_saves_file(self, run_cfg, tmp_path, monkeypatch):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        plotting._save_or_show(fig, "test_plot.png", run_cfg)
        assert (tmp_path / "test_plot.png").exists()

    def test_no_save_when_disabled(self, tmp_path, monkeypatch):
        ...
```
Adapt directly: one `TestXxx` class per plot function, at minimum a `test_does_not_crash` (calls
the function on the fixture, asserts `isinstance(fig, plt.Figure)`) and a `test_empty_input`
(calls on an empty `pd.DataFrame()`/`pd.Series(dtype=float)`, asserts no exception — matching
"does-not-crash + empty-input edge cases" per VALIDATION.md Wave 0 requirements verbatim).

### `tests/unit/test_platform_notebooks.py` (test, static-only — D-17)

No direct analog exists (this is a genuinely new test shape — D-17 explicitly rejects
`nbmake`/`papermill` execution). Build it from `nbformat`'s parse API plus a grep-based import/
plotting-library check:
```python
import nbformat
import json
from pathlib import Path

NOTEBOOK_DIR = Path("notebooks/platform")

def _load(nb_path: Path):
    return nbformat.read(nb_path, as_version=4)

@pytest.mark.parametrize("nb_path", sorted(NOTEBOOK_DIR.glob("*.ipynb")))
def test_notebook_is_valid_json_and_nbformat(nb_path):
    nb = _load(nb_path)  # raises on malformed JSON / bad nbformat
    assert nb.cells

@pytest.mark.parametrize("nb_path", sorted(NOTEBOOK_DIR.glob("*.ipynb")))
def test_notebook_has_no_forbidden_plotting_imports(nb_path):
    nb = _load(nb_path)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        assert "import matplotlib" not in cell.source
        assert "import seaborn" not in cell.source
        assert ".save(" not in cell.source  # D-10: notebooks never write checkpoints
```
Criterion 3 (no inline `matplotlib`/`seaborn`) and the `.save(` checkpoint-write grep (flagged as
a Tampering risk in RESEARCH.md's Security Domain section) are both explicit VALIDATION.md/
RESEARCH.md requirements — include both.

### `notebooks/platform/*.ipynb` (all six — app-tier notebooks)

**Analog:** the 12 legacy notebooks under `notebooks/` (e.g. `notebooks/01_ingestion.ipynb`,
read in full this session — 27 cells, `nbformat` 4/5).

**Confirmed storage convention (do the same for platform notebooks):**
- Stored as plain committed `.ipynb` JSON — **outputs ARE committed** (`test_notebook.cells[i].outputs`
  is non-empty for executed cells; verified directly).
- Cell `id` fields present (e.g. `"id": "md-title"`, `"id": "code-setup"`) — nbformat 4.5+ requires
  these; using `nbformat.v4.new_code_cell(...)`/`new_markdown_cell(...)` auto-generates them.
- First markdown cell is a title + one-paragraph purpose statement + an explicit prerequisite
  instruction (legacy: `"**Run `python pipelines/01_ingest.py` before executing this notebook.**"`).
  Platform equivalent per D-10: state the exact `load_or_explain`-style remediation, e.g.
  *"Run `python scripts/build_platform_data.py` first if `monthly_features` is missing."*
- First code cell is a setup cell: `%matplotlib inline`, `sys.path` shim (legacy) — **platform
  notebooks should NOT need the `sys.path.insert` shim** if the package is `pip install -e`'d;
  confirm the dev-install convention before copying that line — otherwise import directly:
  ```python
  %matplotlib inline
  import logging
  from trading_crab_lib.platform.config import load_platform_config
  from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
  from trading_crab_lib.platform.honesty.holdout import load_full_span
  from trading_crab_lib.platform import plotting as pplot
  cfg = load_platform_config()
  ```
- **What must differ from the legacy pattern:** no `RunConfig(...)` construction (doesn't exist);
  every plot call passes `save_path=...`/`show=False` explicitly instead of a `run_cfg` object;
  and per D-06, any cell reading beyond 2020-12 must call `load_full_span(name)` explicitly rather
  than the plain checkpoint manager `.load(name)` (which is fenced to dev/pre-2021 by construction)
  — this opt-in call should be visually obvious in the notebook (its own cell, with a markdown
  cell immediately above stating why the fence is being crossed, per D-06's honesty-discipline
  framing).

**Author notebooks via `nbformat`'s programmatic API** (per RESEARCH.md Standard Stack —
already installed, 5.11.1, zero new dependency):
```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# P1 — Data Spine\n\n..."),
    nbf.v4.new_code_cell("%matplotlib inline\nimport ..."),
    ...
]
nbf.write(nb, "notebooks/platform/P1_data_spine.ipynb")
```
Do NOT hand-write raw `.ipynb` JSON (error-prone) and do NOT introduce `jupytext` (confirmed not
installed, would be a new dependency creating a second notebook-authoring convention).

### `notebooks/platform/P3_regime_labeling.ipynb` — sign-off cell (D-15)

No analog — this is new, deliberately minimal per D-15 ("no `sign_off()` helper, no YAML
ledger"). Add as the final markdown cell, pre-filled with blank fields for the operator to edit
and re-save:
```markdown
## Cold-Start Sign-Off

**Date:**
**Verdict:** (accept / reject / accept-with-caveats)
**Reasoning:**

(Per D-16: a negative verdict is recorded and does not block the phase.)
```

### `platform/evaluation/report.py` (MODIFIED — additive only, Amendment 3 item H)

**Analog:** the file's own existing `write_backtest_report()` function (lines 306-345) and its
call site inside `run_full_backtest_evaluation()` (lines 660-669) — this is a same-file,
same-pattern addition, not a new pattern import.

**Exact existing artifact-write call to extend** (lines 660-669, quoted verbatim):
```python
kpi_table = _build_kpi_table(strategy_kpis, ablation_kpis, baseline_kpis)
report_path = write_backtest_report(
    markdown,
    {
        "equity_curve_strategy": equity_curve,
        "equity_curve_ablation": ablation_curve,
        "kpi_table": kpi_table,
    },
    output_dir=output_dir,
)
```
`write_backtest_report()`'s own body (lines 332-345) shows exactly how each dict key becomes a
file: `target_dir / f"backtest_{name}.parquet"`, written via `df.to_parquet(artifact_path, index=True)`.

**The additive change (Amendment 3 item H):** add two more keys to this same `artifacts` dict —
`full_sample_states` (already computed in-function at line 593:
`full_sample_states = pd.Series(smoothed_states_arr, index=X_df.index, name="state")`) and a new
per-date filtered-state artifact (built from `per_step_metrics["dates"]`/`per_step_metrics["proba"]`,
already computed at lines 595-611). Example additive diff shape (illustrative, not literal):
```python
report_path = write_backtest_report(
    markdown,
    {
        "equity_curve_strategy": equity_curve,
        "equity_curve_ablation": ablation_curve,
        "kpi_table": kpi_table,
        "full_sample_states": full_sample_states.to_frame(),          # NEW
        "filtered_states_per_date": _build_filtered_state_df(...),    # NEW
    },
    output_dir=output_dir,
)
```
**Constraints that make this a pattern-match, not a free rewrite:**
- Every existing artifact key/filename/schema must be byte-identical after the change —
  `write_backtest_report()` itself needs zero modification; only the caller's `artifacts` dict
  grows two keys. Do not touch `write_backtest_report()`'s signature, its `target_dir` logic, or
  its `df.to_parquet(..., index=True)` call.
- No new computation — `full_sample_states` and the filtered-path data already exist as local
  variables inside `run_full_backtest_evaluation()`; this is purely "also pass it to the function
  that already writes things," per Amendment 3's own framing ("It matches the codebase's existing
  pattern... that function is already the place where evaluation artifacts get persisted").
- Update the function's own docstring return-contract line (`Returns: dict with keys ...
  "full_sample_states"`) — it already lists `full_sample_states` in the returned dict (line 508)
  but that dict entry and the *persisted* artifact are currently two different things; verify
  during implementation whether `full_sample_states` is already being returned (line ~671-679,
  confirmed: yes, `"full_sample_states": full_sample_states` likely appears near the return dict)
  — if so, this task is "also parquet-persist what's already returned," the smallest possible
  version of the change.

## Shared Patterns

### No-`RunConfig` plot signature (binding on every plotting file)
**Source:** `platform/config.py:56-94` (proves no RunConfig-shaped object exists) + CONTEXT.md D-02
**Apply to:** every function in `platform/plotting/{core,data,features,regime,nowcaster,allocation,backtest,drift}.py`
```python
def plot_x(data: pd.DataFrame, *, save_path: Path | None = None, show: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(...))
    ...
    return _save_or_show(fig, save_path=save_path, show=show)
```

### Matplotlib Agg-backend guard (verbatim, copy from legacy)
**Source:** `trading_crab_lib/plotting/core.py:31-53`
**Apply to:** `platform/plotting/core.py` only (imported/used by all other plotting submodules
transitively through `core._save_or_show`/`core._in_jupyter`)

### Plausibility raise/warn idiom (D-11)
**Source:** `platform/splice.py:124-155` (`assert_yield_units_plausible`)
**Apply to:** every band function in `platform/plotting/drift.py`

### Checkpoint load-or-raise-actionably (D-10)
**Source:** derived pattern, no existing helper to import — build fresh per RESEARCH.md Pattern 2,
wrapping `CheckpointManager.load()`'s existing `FileNotFoundError` contract
(`src/trading_crab_lib/checkpoints.py`, and platform's own `platform/checkpoints.py::get_platform_checkpoint_manager()`)
**Apply to:** the first data-loading cell of every one of the six notebooks
```python
def load_or_explain(name: str, *, rebuild_hint: str) -> pd.DataFrame:
    try:
        return get_platform_checkpoint_manager().load(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Checkpoint '{name}' not found in data/checkpoints/platform/. Run: {rebuild_hint}"
        ) from exc
```
Discretion item: this helper can live in `platform/plotting/core.py` or a new
`notebooks/platform/_utils.py` — either satisfies D-10; RESEARCH.md leans toward the plotting
package for testability (ADR #11's "plotting logic never inline" doctrine extends naturally to
"loading logic never inline" here too), which this pattern map recommends without locking it.

### Full-span opt-in (D-06) — never open two managers by hand
**Source:** `platform/honesty/holdout.py::load_full_span(name)` (already implemented; per
CONTEXT.md AMENDMENT item B, do not hand-roll a dual-manager concatenation)
```python
from trading_crab_lib.platform.honesty.holdout import load_full_span
full_features = load_full_span("monthly_features")  # 776 rows through 2026-08, D-06 opt-in
```

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `platform/plotting/drift.py`'s D-09 drift statistic function | utility | transform | Genuinely new logic (Claude's Discretion on exact statistic); nearest precedent is the *shape* of plausibility checks (`assert_yield_units_plausible`), not the drift math itself — no existing drift-against-baseline computation exists anywhere in the codebase |
| `tests/unit/test_platform_notebooks.py`'s static-parse-and-grep approach | test | transform | D-17 deliberately rejects the one analog that would otherwise apply (`nbmake`/execution-based notebook testing); built instead from `nbformat`'s parse API + plain string grep, which has no existing test-file precedent in this repo |
| P3's sign-off markdown cell content | notebook content | n/a | D-15 explicitly forbids porting/inventing a `sign_off()` helper; the cell is prose only |

## Metadata

**Analog search scope:** `src/trading_crab_lib/plotting/` (9 files, full read of `core.py` +
`__init__.py`), `src/trading_crab_lib/platform/{splice.py,config.py,evaluation/report.py,
checkpoints.py,honesty/holdout.py}`, `tests/unit/test_plotting.py`, `tests/integration/
test_mini_backtest.py` (referenced via RESEARCH.md), `notebooks/01_ingestion.ipynb` (full JSON
structure read).
**Files scanned:** ~10 read in full or substantially; cross-referenced against 06-RESEARCH.md's
own verified line citations (all claims in RESEARCH.md were themselves read from source this
session per its Sources section — used as a corroborating cross-check, not the sole basis).
**Pattern extraction date:** 2026-09-09

## PATTERN MAPPING COMPLETE

**Phase:** 6 - Platform Notebook Suite
**Files classified:** ~19 (9 plotting modules, 6 notebooks, 3 test files, 1 modified library file)
**Analogs found:** 15 / 15 (3 items have "No Analog Found" — all deliberately, per locked decisions D-15/D-17/D-09)

### Coverage
- Files with exact analog (pattern source confirmed, signature must diverge): 3 (`core.py`, `__init__.py`, `test_platform_plotting.py`)
- Files with role-match analog: 12 (per-layer plotting modules, 6 notebooks, `test_platform_plotting_drift.py`, modified `report.py`)
- Files with no analog: 3 (drift statistic, static-notebook test shape, sign-off cell)

### Key Patterns Identified
- **The platform plotting package must NOT take a `RunConfig`.** Verified `platform/config.py`
  has no such object; the correct signature is `plot_x(data, *, save_path: Path | None = None,
  show: bool = False) -> Figure`, confirmed against both CONTEXT.md D-02 and a direct read of
  `platform/config.py:56-94`.
- **`_save_or_show`, `_in_jupyter`, `CUSTOM_COLORS`/`REGIME_CMAP`, and the matplotlib
  Agg-backend guard are copy-the-idiom-not-the-import** (D-01) from `trading_crab_lib/plotting/core.py`.
- **D-11 plausibility bands must follow `assert_yield_units_plausible`'s raise/warn shape**
  (`platform/splice.py:124-155`) — named constants, hard `ValueError` with an explanatory
  f-string, `log.warning` only for the ambiguous direction. Two named regression cases (60/40 at
  −2.27% DD, terminal log wealth 111.06) are non-negotiable test content per VALIDATION.md.
- **The Amendment 3 additive write to `platform/evaluation/report.py` is a same-file pattern
  extension**, not a new pattern — extend the existing `artifacts` dict passed into
  `write_backtest_report()` (lines 660-669) with two new keys; `write_backtest_report()` itself
  needs no changes, and every existing artifact must remain byte-identical.
- **Notebooks are committed with outputs, use `nbformat`'s programmatic API (not jupytext or
  hand JSON), and open with a title/prereq markdown cell + a setup code cell** — confirmed
  directly from `notebooks/01_ingestion.ipynb`'s JSON structure.

### File Created
`/home/user/claude-scratch-work/.planning/phases/06-platform-notebook-suite/06-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can now reference analog patterns in PLAN.md files.
