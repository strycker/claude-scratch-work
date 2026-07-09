# Phase E Plan — Structural Cleanup & Refactor

> **⚠️ SUPERSEDED (July 2026).** Historical plan — kept for context, not active work.
> The project's target and execution plan are now `platform_design/platform_design.md`
> (v1.7) and `ROADMAP.md` Tier 0. Do not treat items below as current.

**Depends on:** Phase C (2-package infrastructure) — merged as PR #67.
**Branch:** `claude/implement-phase-a2-SmHEA`

---

## E1 — Update MANIFEST.in

**Goal:** Ensure both `trading-crab` (app) and `trading-crab-lib` (library) sdists
include exactly the right files and exclude everything else.

### Root `MANIFEST.in` (app sdist)
- Include `src/trading_crab/` files only (not `trading_crab_lib/`)
- Exclude `run_pipeline.py` (kept as backward-compat shim, not shipped in app sdist)
- Prune submodules, notebooks, scripts, data, outputs
- Keep LICENSE, README.md, pyproject.toml

### `src/trading_crab_lib/MANIFEST.in` (library sdist)
- Include `trading_crab_lib/` recursive Python files + `py.typed`
- Exclude config, data, outputs, notebooks, scripts
- Global-exclude `.env`, `*.pyc`, `__pycache__`, `.DS_Store`

---

## E2 — Update CLAUDE.md

**Goal:** Reflect the 2-package architecture in the project guide.

### Changes:
1. Add `src/trading_crab/` to the repository layout tree
2. Add a "Two-Package Architecture" section explaining `trading-crab` vs `trading-crab-lib`
3. Update "How to Run" with `tradingcrab` CLI entry point examples
4. Update "Environment Setup" with `pip install -e .` and `pip install -e src/trading_crab_lib/[all]`
5. Note `uv sync` as workspace-aware alternative
6. Update "Key Design Decisions" to mention the app/lib split

---

## E3 — Update README.md

**Goal:** Make README.md reflect current project state for new users.

### Changes:
1. Add "Two Packages" explanation near the top
2. Update Installation section: `pip install trading-crab` (app+CLI) vs `pip install trading-crab-lib` (library only)
3. Update layout diagram to show `src/trading_crab/`
4. Add PyPI badges placeholder (for when packages are published)
5. Remove references to `requirements.txt` / `requirements-dev.txt` as primary install method (keep as fallback mention)
6. Update dependency table to reference optional extras

---

## E4 — Refactor `run_pipeline.py` → `src/trading_crab/pipeline.py`

**Goal:** Move pipeline orchestration logic into the app package so `tradingcrab` CLI
works without sys.path hacks.

### Approach:
1. Create `src/trading_crab/pipeline.py` with all pipeline logic from `run_pipeline.py`
2. Move `build_parser()`, `main()`, all `step*()` functions, and `STEPS` dict
3. Update imports: all `trading_crab_lib.*` imports stay as-is; path constants
   (`ROOT`, `DATA_DIR`, etc.) come from `trading_crab_lib`
4. `run_pipeline.py` becomes a thin 5-line shim:
   ```python
   """Backward-compatible entry point. Use `tradingcrab` CLI instead."""
   from trading_crab.pipeline import main
   if __name__ == "__main__":
       main()
   ```
5. `src/trading_crab/cli.py` simplified:
   ```python
   def run_pipeline():
       from trading_crab.pipeline import main
       main()
   ```
6. All tests that import from `run_pipeline` continue working via the shim

### Risk mitigation:
- Keep `run_pipeline.py` as a working shim (no breakage for existing users)
- All step functions remain importable from `trading_crab.pipeline`
- `pipelines/*.py` standalone scripts are NOT moved (they're separate entry points)

---

## Phase F — Re-planned (post-E)

Phase F from NEXT_STEPS.md was too broad. Here's a more granular breakdown,
designed for incremental migration to the human-verified `strycker/trading-crab` repo.

### F1 — Library config independence (S)
Allow `trading-crab-lib` to accept a config dict or path at runtime rather than
requiring `config/settings.yaml` on disk. Enables clean `pip install` usage.

**Questions for user:**
- Should `load()` accept an optional `config_path` kwarg?
- Should there be a `load_from_dict(d)` alternative?
- What's the minimum config a library user needs to call `engineer_all()`?

### F2 — Backtest framework (XL)
Walk-forward quarterly backtesting: retrain model each quarter, predict regime,
construct portfolio, measure returns vs benchmarks.

**Questions for user:**
- Which benchmarks? SPY, 60/40, equal-weight?
- Retrain from scratch each quarter or incremental?
- Should this be a library module or a pipeline step?

### F3 — Per-asset probability models (L)
Binary classifiers per ETF: "will this asset be +X% at Y quarters?"

**Questions for user:**
- Which ETFs to start with? Just SPY, TLT, GLD?
- What thresholds for X? 5%, 10%, 20%?
- Should these use regime as a feature or be regime-independent?

### F4 — Streamlit dashboard (L)
Interactive browser-based UI for regime history, asset rankings, portfolio.

**Questions for user:**
- Priority vs email reports?
- Deploy locally or host somewhere?
- Which views are most important first?

### F5 — AI narrative weekly report (XL)
Claude API generates market commentary from regime data.

### F6 — Finviz Elite integration (L)
Sector rotation, institutional flow data. Gated behind API key.

**Recommended order:** F1 → F2 → F3 → F4 → F5 → F6
F1 is the prerequisite for clean library usage in the human-verified repo.

---

## Migration Strategy (post-E)

The user's goal after Phase E is to incrementally migrate pieces to `strycker/trading-crab`.

**Recommended migration order:**
1. `trading-crab-lib` package (self-contained, no app dependencies)
2. `config/settings.yaml` (after F1 makes it optional)
3. `trading-crab` app package + `run_pipeline.py` shim
4. Tests (after both packages are in the target repo)
5. Notebooks (last — they're exploration tools)

Each piece should be independently testable in the target repo before moving the next.
