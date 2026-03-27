# Rename Plan: `market_regime` → `trading_crab_lib`

## Scope

Rename the Python package from `market_regime` to `trading_crab_lib` across the entire
codebase. **438 references across 89 files** (excluding read-only submodules).

---

## Pre-Conditions

- All tests pass before starting
- Clean git working tree
- Submodules (`gsd-scratch-work/`, `trading-crab-lib/`) are untouched

---

## Phase 1: Package Directory Rename (highest risk, do first)

### Step 1.1 — Rename `src/market_regime/` → `src/trading_crab_lib/`
```
mv src/market_regime src/trading_crab_lib
```
This is the atomic operation that everything else follows from.

### Step 1.2 — Delete generated `src/market_regime.egg-info/`
```
rm -rf src/market_regime.egg-info
```
Will be regenerated as `src/trading_crab_lib.egg-info` after `pip install -e .`

### Step 1.3 — Update `pyproject.toml`
- `name = "market-regime"` → `name = "trading-crab"`
- `market_regime = ["py.typed"]` → `trading_crab_lib = ["py.typed"]`
- Any other references in build config

### Step 1.4 — Update `MANIFEST.in`
- `recursive-include src/market_regime py.typed` → `recursive-include src/trading_crab_lib py.typed`

### Step 1.5 — Reinstall in editable mode
```
pip install -e ".[dev]"
```
Verify: `python -c "import trading_crab_lib; print(trading_crab_lib.__file__)"`

---

## Phase 2: Source Code Imports (26 files)

Update all intra-package imports within `src/trading_crab_lib/`:

- `__init__.py` — module docstring
- `config.py` — any self-references
- `checkpoints.py` — 2 occurrences
- `transforms.py` — 2 occurrences
- `plotting.py` — 5 occurrences (imports from other market_regime modules)
- `regime.py` — 2 occurrences
- `asset_returns.py` — 2 occurrences
- All other modules — 1 occurrence each (typically docstring or internal import)
- `ingestion/__init__.py`, `ingestion/assets.py`, `ingestion/grok.py`, `ingestion/macrotrends.py`
- `prediction/__init__.py`, `prediction/classifier.py`, `prediction/gradient_boosting.py`

**Approach:** Global find-and-replace `from market_regime` → `from trading_crab_lib` and
`import market_regime` → `import trading_crab_lib` across all `.py` files in `src/trading_crab_lib/`.

---

## Phase 3: Pipeline & Script Imports (11 files)

- `run_pipeline.py` — ~35 occurrences (largest consumer)
- `pipelines/01_ingest.py` through `pipelines/09_tactics.py` — 9 files
- `scripts/run_weekly_report.py` — 2 occurrences

Same pattern: `from market_regime.X import Y` → `from trading_crab_lib.X import Y`.

---

## Phase 4: Test Files (31 files)

All test files under `tests/` and `tests/unit/`:

- `test_ingestion.py` — 39 occurrences (monkeypatch paths like `"market_regime.ingestion.multpl.requests"`)
- All other test files — 1–9 occurrences each

**Critical:** monkeypatch targets like `monkeypatch.setattr("market_regime.ingestion.fred.Fred", ...)`
must be updated to `"trading_crab_lib.ingestion.fred.Fred"`. These are string references,
not import statements — a simple find-replace handles them.

---

## Phase 5: Jupyter Notebooks (8 files)

- 51 occurrences across 8 notebooks in `notebooks/`
- All in import cells or inline comments
- `notebooks/03_clustering.ipynb` has 17 occurrences (largest)

**Approach:** Use `sed` or a notebook-aware tool to replace in `.ipynb` JSON source cells.

---

## Phase 6: Documentation (4 files)

- `CLAUDE.md` — ~13 occurrences (layout tree, import examples, ADRs, pitfalls)
- `README.md` — ~2 occurrences
- `STATE.md` — ~6 occurrences
- `ROADMAP.md` — ~20 occurrences

Also update:
- `src/market_regime/` → `src/trading_crab_lib/` in all directory references
- `market_regime.prediction` → `trading_crab_lib.prediction` in code examples
- Keep references to `market_code` column name unchanged (that's a data column, not the package)

---

## Phase 7: Config Files

- `config/settings.yaml` — ~4 occurrences (if any reference the package name vs data columns)

**IMPORTANT:** `market_code` is a data column name (per-quarter regime label), NOT a
package reference. Do NOT rename `market_code` → anything. Only rename references to
the `market_regime` Python package.

---

## Phase 8: Verification

### Step 8.1 — Grep for stragglers
```bash
grep -r "market_regime" --include="*.py" --include="*.md" --include="*.yaml" \
  --include="*.toml" --include="*.in" --include="*.ipynb" \
  --exclude-dir=gsd-scratch-work \
  --exclude-dir=trading-crab-lib \
  --exclude-dir=legacy
```
(Note: `legacy/` should NOT be modified per project rules.)

### Step 8.2 — Run full test suite
```bash
pytest tests/ -v
```

### Step 8.3 — Verify pipeline import chain
```bash
python -c "from trading_crab_lib.config import load; print(load()['data'])"
python -c "from trading_crab_lib.prediction import train_current_regime; print('OK')"
python -c "from trading_crab_lib.checkpoints import CheckpointManager; print(CheckpointManager().list())"
```

### Step 8.4 — Verify editable install
```bash
pip show trading-crab
python -c "import trading_crab_lib; print(trading_crab_lib.__file__)"
```

---

## Things NOT to Rename

- `market_code` — This is a DataFrame column name (per-quarter regime label integer).
  Appears in settings.yaml, transforms.py, run_pipeline.py CLI flags (`--market-code`),
  and throughout the pipeline. It is NOT related to the package name.
- Files in `legacy/` — Never modify. Per CLAUDE.md rules.
- Files in `gsd-scratch-work/` or `trading-crab-lib/` — Read-only submodules.
- `market_regime` in git commit history — historical, cannot be changed.

---

## Estimated Effort

| Phase | Files | Est. Time |
|-------|-------|-----------|
| 1. Directory + build config | 4 | Quick |
| 2. Source code imports | 26 | Moderate (careful with intra-package refs) |
| 3. Pipeline/script imports | 11 | Quick (uniform pattern) |
| 4. Test files | 31 | Moderate (monkeypatch strings need care) |
| 5. Notebooks | 8 | Quick (JSON string replace) |
| 6. Documentation | 4 | Moderate (context-sensitive) |
| 7. Config | 1-2 | Quick |
| 8. Verification | — | Required |

**Total: ~89 files, ~438 replacements. Single focused session.**

---

## Commit Strategy

One commit per phase (or one atomic commit for the whole rename):

- Option A: **Single atomic commit** — easier to revert if something breaks
- Option B: **Per-phase commits** — easier to review, but intermediate states may not pass tests

**Recommendation: Option A** — single atomic commit `refactor: rename market_regime → trading_crab_lib`
since intermediate states (e.g., directory renamed but imports not updated) will break everything.
