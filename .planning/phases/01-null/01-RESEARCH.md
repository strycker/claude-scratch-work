# Phase 1: Data & Constraints Foundations - Research

**Researched:** 2026-03-16  
**Domain:** Data ingestion, checkpointing, feature engineering, and constraints enforcement for Trading-Crab  
**Confidence:** HIGH

## User Constraints

### Locked Decisions
- ETF-only universe for v1 (no single stocks, no direct crypto).
- Bitcoin exposure only via ETF wrappers.
- Weekly report cadence; regimes at quarterly resolution.

### Claude's Discretion
- How to structure the concrete work inside Phase 1 to satisfy DATA-01/02/03 and CONSTR-01/02 using the existing `src/market_regime/` and `pipelines/` stack.
- Which additional config surfacing, validation checks, and tests to add (as long as they respect the ETF-only, non-intraday, non-auto-trading constraints).

### Deferred Ideas (OUT OF SCOPE)
- Any extension beyond ETF-only (single stocks, direct crypto, options, leverage, auto-execution) is explicitly out of scope for v1 and should not influence Phase 1 design.

## Summary

Phase 1 does not need to invent a new data stack; it needs to **productize and harden** the existing ingestion, checkpointing, and feature engineering pipeline in `src/market_regime/` so that it cleanly satisfies DATA-01/02/03 and CONSTR-01/02. The project already has working ingestion for multpl.com and FRED (`ingestion.multpl`, `ingestion.fred`), ETF price ingestion with a robust fallback chain (`ingestion.assets`), checkpoint management (`io.checkpoints.CheckpointManager`), a feature pipeline with causal/non-causal variants (`features.transforms.engineer_all(causal=...)`), and a CLI-oriented runner (`run_pipeline.py` plus `pipelines/01–07`). Phase 1’s planning should treat these as the **standard stack to align and surface**, not as optional helpers.

Planning this phase well means: (1) making the data universe and constraints explicit in config (`config/settings.yaml` and possibly a small ETF-universe manifest), (2) ensuring that ingestion and checkpoint behavior for macro + ETFs is observable, reproducible, and cheap to iterate on, (3) clearly documenting and surfacing the feature set (including causal variants) as a stable contract for downstream phases, and (4) baking ETF-only / non-intraday / non-auto-trading constraints into both code paths and tests so regressions are hard. The key risk is not “can we ingest or compute features?” (we can) but “will downstream work accidentally violate constraints or subtly depend on fragile, under-documented data behavior?”.

**Primary recommendation:** Use the existing `run_pipeline.py` + `CheckpointManager` + `engineer_all(causal=True/False)` stack as the canonical Phase 1 foundation, and plan work around **making configuration, invariants, and tests explicit** rather than re-implementing ingestion or features.

## Standard Stack

### Core

| Library / Module | Version | Purpose | Why Standard |
|------------------|---------|---------|--------------|
| Python 3.10+ | N/A | Language/runtime | Project conventions, type-hint style, and existing codebase all target 3.10+. |
| `pandas` | As pinned in `pyproject.toml` | DataFrame manipulation, resampling to quarterly | Underpins all time-series handling; already used across ingestion and features. |
| `pyarrow` | As pinned | Parquet I/O for checkpoints | Chosen in `CLAUDE.md` as standard for DataFrame persistence; used by `CheckpointManager`. |
| `fredapi` | As pinned | FRED macro series ingestion | Current implementation in `ingestion.fred.fetch_all(cfg)` is built on this. |
| `requests` + `lxml` | As pinned | multpl.com scraping | `ingestion.multpl.fetch_all(cfg)` already matches the legacy behavior. |
| `yfinance` (+ optional `curl_cffi`) | As pinned | ETF monthly prices → quarterly | `ingestion.assets` is the canonical ETF price stack with multi-step fallback and SSL handling. |
| `scikit-learn` | As pinned | PCA, derivatives of feature work, later clustering/models | Already wired into features + later phases; Phase 1 should respect current usage, not change its role. |
| `market_regime.io.checkpoints.CheckpointManager` | Local | Parquet checkpoints + manifest | Core abstraction for DATA-02; all pipeline steps should standardize on this. |
| `market_regime.features.transforms.engineer_all` | Local | End-to-end feature engineering, including causal variants | Encodes the required feature pipeline and is explicitly called out in `CLAUDE.md`. |

### Supporting

| Library / Module | Version | Purpose | When to Use |
|------------------|---------|---------|-------------|
| `pathlib` | Stdlib | File paths for data/checkpoints/outputs | Anywhere paths are handled (Phase 1 should avoid string concatenation). |
| `logging` | Stdlib | Structured logs for ingestion/feature runs | All ingestion + feature modules already use `logging`; Phase 1 plans should rely on and, if needed, extend this. |
| `run_pipeline.py` | Local | Unified CLI for pipeline steps and flags | Primary entrypoint for “end-to-end” behaviors in Phase 1 (with `--steps`/`--refresh`/`--recompute`). |
| `pipelines/01_ingest.py`, `02_features.py` | Local | Step-wise orchestration for ingestion & features | Phase 1 planning should ensure these scripts are in sync with requirements and checkpoints. |
| `config/settings.yaml` | Local | Central configuration for data/feature/ingestion settings | All data-universe and feature-related knobs should live here, not be hard-coded. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom file-based checkpoint hacks | `CheckpointManager` | Centralized manifest, path conventions, and freshness logic already exist; re-implementing risks divergence and subtle bugs. |
| Ad-hoc scripts for multpl/FRED/ETF ingestion | Existing `ingestion.*` modules | The legacy-aligned ingestion modules already encapsulate URL formats, rate limits, publication-lag shifts, and fallbacks. Replacing them would add risk without value. |
| Feature engineering hand-coded in notebooks | `features.transforms.engineer_all` | Notebooks are for exploration; Phase 1 should treat `engineer_all` as the single source of truth for features. |

**Installation (developer environment):**

```bash
pip install -e ".[dev]"
```

## Architecture Patterns

### Recommended Project Structure (Phase-1-Relevant)

```text
config/
  settings.yaml          # Data universe, FRED/multpl configs, feature lists

data/
  raw/                   # macro_raw.parquet, asset_prices.parquet
  processed/             # features.parquet (and causal variant)
  regimes/               # downstream (not Phase 1’s focus)
  checkpoints/           # parquet checkpoints managed by CheckpointManager

pipelines/
  01_ingest.py           # Orchestrates macro + ETF ingestion + checkpoints
  02_features.py         # Orchestrates feature engineering (causal + non-causal)

src/market_regime/
  config.py              # load() for settings, setup_logging()
  runtime.py             # RunConfig (refresh/recompute/plots, etc.)
  io/checkpoints.py      # CheckpointManager
  ingestion/
    fred.py              # FRED.fetch_all(cfg)
    multpl.py            # multpl.fetch_all(cfg)
    assets.py            # ETF price fetch with fallback chain
  features/
    transforms.py        # engineer_all(causal=...), feature contracts
```

Phase 1 planning should assume this structure is **fixed** and design tasks that:
- Make configuration and contracts (inputs/outputs of each step) explicit.
- Ensure `01_ingest.py` and `02_features.py` consistently use `RunConfig` + `CheckpointManager`.
- Emit and document artifacts (parquet files, logs) that serve as inputs to later phases.

### Pattern 1: Config-driven ingestion and features

**What:** All ingestion and feature behavior should be driven by `config/settings.yaml` plus a small, explicit ETF-universe list, not by hard-coded constants in Python.

**When to use:** Any time Phase 1 needs to adjust data coverage (new ETF tickers, new FRED series, yield-curve features) or the feature set; modifications should be applied via config, not code.

**Example (conceptual):**

```python
from market_regime.config import load
from market_regime.ingestion import fred, multpl, assets
from market_regime.io.checkpoints import CheckpointManager

cfg = load()
cm = CheckpointManager()

if not cm.is_fresh("macro_raw"):
    macro = fred.fetch_all(cfg)
    macro_multpl = multpl.fetch_all(cfg)
    cm.save_parquet("macro_raw", macro.join(macro_multpl, how="outer"))

if not cm.is_fresh("asset_prices"):
    prices = assets.fetch_all(cfg)
    cm.save_parquet("asset_prices", prices)
```

### Pattern 2: Dual feature outputs (causal vs non-causal)

**What:** Use `features.transforms.engineer_all(causal=False)` for exploratory/unsupervised work, and `engineer_all(causal=True)` for any features that will feed supervised models, writing both to clear, separate parquet outputs.

**When to use:** In Phase 1’s feature orchestration (likely `pipelines/02_features.py`), whenever features are recomputed from raw/checkpointed data.

**Example (conceptual):**

```python
from market_regime.config import load
from market_regime.features.transforms import engineer_all
from market_regime.io.checkpoints import CheckpointManager

cfg = load()
cm = CheckpointManager()

features_noncausal = engineer_all(cfg, causal=False)
features_causal = engineer_all(cfg, causal=True)

cm.save_parquet("features_noncausal", features_noncausal)
cm.save_parquet("features_causal", features_causal)
```

### Anti-Patterns to Avoid

- **Hard-coding data universe or URLs in Python:** All multpl/FRED datasets, ETF tickers, and feature lists must live in `settings.yaml` (or a small ETF manifest), not in pipeline scripts. This keeps Phase 1 in sync with config-based design.
- **Bypassing `CheckpointManager` with ad-hoc file saves:** Writing raw/processed parquet files outside `data/checkpoints/` without manifest/freshness tracking will confuse later steps and make DATA-02 brittle.
- **Mixing causal and non-causal features in a single, unlabeled artifact:** Downstream phases need to know exactly which feature set they’re using; Phase 1 should never produce an ambiguous “features.parquet” that blends causal/non-causal logic.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Macro series ingestion (FRED) | Custom HTTP/API clients | `ingestion.fred.fetch_all(cfg)` | Already handles quarterly resampling, publication-lag shifts, and config-driven series definitions. |
| multpl.com scraping | New scrapers or parsing logic | `ingestion.multpl.fetch_all(cfg)` | Matches legacy behavior, handles units/percents, and enforces polite rate limiting. |
| ETF price ingestion & fallbacks | Custom yfinance wrappers or alternate sources | `ingestion.assets.fetch_all(cfg)` | Encapsulates multi-step fallback chain, SSL workarounds, and quarterly resampling. |
| Checkpointing / freshness logic | Ad-hoc timestamp files or direct parquet writes | `io.checkpoints.CheckpointManager` | Centralized manifest, consistent naming, and CLI integration already exist. |
| Feature engineering pipeline | Notebook-only transformations | `features.transforms.engineer_all` | Implements the exact, ordered pipeline required by `CLAUDE.md`, including gap-filling and derivatives. |

**Key insight:** Phase 1’s job is to standardize and expose **contracts and configuration** for the existing stack, not to design new ingestion or checkpointing mechanisms from scratch.

## Common Pitfalls

### Pitfall 1: Silent look-ahead leakage

**What goes wrong:** Features or labels accidentally use future information (e.g. unshifted GDP/GNP, non-causal rolling windows) when training supervised models or computing “current” features.

**Why it happens:** It’s easy to reuse the richer non-causal feature set or forget which FRED series are shifted. Without explicit separation and documentation, downstream code may consume the wrong artifact.

**How to avoid:**
- Treat **causal vs non-causal features as separate, named artifacts** with clear filenames and documentation.
- Ensure that any FRED series marked `shift: true` in config remain shifted in `ingestion.fred`.
- Document in Phase 1 outputs that supervised models MUST use the causal feature parquet.

**Warning signs:**
- Feature sets used for supervised training and clustering are loaded from the same file without clear causal flag.
- Model performance appears suspiciously high relative to out-of-sample expectations.

### Pitfall 2: Violating ETF-only / non-intraday constraints

**What goes wrong:** Additional assets or data frequencies creep into the codebase or config (single stocks, intraday prices, direct crypto), violating CONSTR-01/02.

**Why it happens:** Experimentation in notebooks or pipeline tweaks may add tickers or daily data that aren’t formally constrained.

**How to avoid:**
- Maintain a **single source of truth** for the ETF universe (e.g. `settings.yaml` `assets.etfs` list) and ensure ingestion scripts derive their tickers exclusively from it.
- Keep all pipelines at monthly/quarterly resolutions; do not introduce intraday intervals.
- Write tests/assertions that fail if non-ETF tickers or sub-daily frequencies appear in core artifacts.

**Warning signs:**
- Ad-hoc tickers hard-coded in notebooks or pipeline scripts.
- New columns or tickers in checkpointed data that aren’t in the documented ETF universe.

### Pitfall 3: Broken or stale checkpoints

**What goes wrong:** Checkpoints become inconsistent with code or config changes, leading to confusing results or subtle errors when `--recompute`/`--refresh` behavior changes.

**Why it happens:** Checkpoints are not versioned or keyed on config hashes; users may rely on stale data without realizing it.

**How to avoid:**
- Rely on `CheckpointManager`’s manifest and freshness logic, not direct parquet-file inspection.
- Plan tasks to verify and document how config changes (e.g. ETF universe, FRED series) are reflected in checkpoint keys or metadata.
- Provide a clear “reset” or “clear stale checkpoints” command in Phase 1 docs or CLI usage notes.

**Warning signs:**
- Running with new `settings.yaml` but seeing identical outputs without recompute.
- Unexpected differences between runs when `--refresh` vs `--recompute` flags are toggled.

## Code Examples

### Example 1: Using `CheckpointManager` in a pipeline step (conceptual)

```python
from market_regime.config import load, setup_logging
from market_regime.io.checkpoints import CheckpointManager
from market_regime.ingestion import fred, multpl, assets
from market_regime.runtime import RunConfig

def main(run_cfg: RunConfig) -> None:
    cfg = load()
    setup_logging(run_cfg.verbose)
    cm = CheckpointManager()

    if run_cfg.refresh_source_datasets or not cm.is_fresh("macro_raw"):
        macro = fred.fetch_all(cfg)
        multpl_df = multpl.fetch_all(cfg)
        cm.save_parquet("macro_raw", macro.join(multpl_df, how="outer"))

    if run_cfg.refresh_source_datasets or not cm.is_fresh("asset_prices"):
        prices = assets.fetch_all(cfg)
        cm.save_parquet("asset_prices", prices)
```

### Example 2: Generating both causal and non-causal feature sets (conceptual)

```python
from market_regime.config import load
from market_regime.features.transforms import engineer_all
from market_regime.io.checkpoints import CheckpointManager

def compute_features() -> None:
    cfg = load()
    cm = CheckpointManager()

    features = engineer_all(cfg, causal=False)
    features_causal = engineer_all(cfg, causal=True)

    cm.save_parquet("features_noncausal", features)
    cm.save_parquet("features_causal", features_causal)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Monolithic legacy script without standardized checkpoints | Modular `src/market_regime/` + `pipelines/` + `CheckpointManager` | 2025–2026 refactor | Enables Phase 1 to build on a clean, testable architecture instead of rewriting from scratch. |
| Ad-hoc feature engineering scattered across notebooks | Centralized `engineer_all` pipeline with config-driven feature lists | As per `CLAUDE.md` design | Gives Phase 1 a single place to define and document the feature contract. |
| Direct macro-ETF joins without clear causal variants | Dual outputs (causal vs non-causal) controlled by `engineer_all(causal=...)` | Implemented before current STATE.md | Lets Phase 1 enforce causal correctness for supervised learning while preserving rich features for clustering. |

**Deprecated/outdated (for Phase 1):**
- Relying on the legacy `unified_script.py` as an execution path; it is now a **reference only**. All Phase 1 plans should operate on `src/` + `pipelines/`.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Ingest macro series and ETF prices for the configured universe over the intended historical window. | Existing `ingestion.fred`, `ingestion.multpl`, and `ingestion.assets` already implement the necessary ingestion logic; Phase 1 should formalize configuration, ETF-universe definitions, and end-to-end ingestion + checkpoint orchestration via `01_ingest.py` and `CheckpointManager`. |
| DATA-02 | Maintain a checkpointed data pipeline so full re-scrapes are optional. | `CheckpointManager` and `data/checkpoints/` are already the canonical mechanism; Phase 1 should standardize their use in all ingestion/feature steps, document typical `--refresh`/`--recompute` workflows, and ensure manifest/freshness semantics are well understood. |
| DATA-03 | Compute a stable, documented feature set with causal variants. | `features.transforms.engineer_all(causal=...)` and config-driven feature lists in `settings.yaml` provide the core; Phase 1 should make dual parquet outputs (causal/non-causal) and feature documentation explicit and stable. |
| CONSTR-01 | Enforce ETF-only universe (including bitcoin via ETF). | The asset universe is already described in project docs and `settings.yaml`; Phase 1 should ensure ingestion, checkpoints, and features only reference configured ETFs and add tests/guards against non-ETF tickers. |
| CONSTR-02 | No intraday / auto-trading behavior; weekly/quarterly cadence only. | Current ingestion uses monthly→quarterly resampling and no broker integration; Phase 1 should codify this as a constraint (config + tests), ensure no sub-daily intervals are introduced, and keep outputs as recommendations/reports only. |

## Validation Architecture

Nyquist-style validation is **enabled** (`nyquist_validation: true` in `.planning/config.json`), so Phase 1 plans should integrate with the existing Python test stack.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `pytest` (configured via `pyproject.toml`) |
| Config file | `pyproject.toml` pytest section (no separate `pytest.ini` required) |
| Quick run command | `pytest tests/ -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map (initial)

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Ingestion of macro + ETF data over configured window succeeds and produces expected shapes/index frequencies. | integration | `pytest tests/ -k "ingestion" -v` | ✅ (ingestion tests already exist; Phase 1 can extend) |
| DATA-02 | Checkpointed pipeline correctly skips re-scrapes when data is fresh and respects `--refresh`/`--recompute`. | integration | `pytest tests/ -k "checkpoint" -v` | ✅ (CheckpointManager tests exist; Phase 1 may add pipeline-level cases) |
| DATA-03 | Feature engineering produces both non-causal and causal feature sets with no NaNs where they shouldn’t be and correct column contracts. | unit/integration | `pytest tests/ -k "features" -v` | ✅ (feature tests exist; Phase 1 may add causal-specific tests) |
| CONSTR-01 | Core artifacts only contain configured ETF tickers; non-ETF tickers cause failures. | unit/integration | `pytest tests/ -k "assets or etf" -v` | ❌ Wave 0 (likely needs explicit constraint tests) |
| CONSTR-02 | No sub-daily resolutions appear in core artifacts; no auto-trading outputs are produced. | integration | `pytest tests/ -k "constraints" -v` | ❌ Wave 0 (needs targeted tests/assertions) |

### Sampling Rate

- **Per task commit:** Run a focused subset (e.g. `pytest tests/ -k "ingestion or features or checkpoint" -q`) whenever Phase 1 code touching ingestion, features, or checkpoints changes.
- **Per wave merge:** Run `pytest tests/ -v` before marking a Phase 1 planning wave as complete.
- **Phase gate:** Full `pytest tests/ -v` green (including new constraint-focused tests) before `/gsd:verify-work` for Phase 1.

### Wave 0 Gaps (to be addressed by planning)

- [ ] Add explicit tests that enforce ETF-only tickers in key artifacts (e.g. asset price checkpoints, feature sets).
- [ ] Add tests that assert quarterly frequency (and no sub-daily indices) for core data/feature artifacts.
- [ ] Optionally add smoke tests for `pipelines/01_ingest.py` and `02_features.py` to ensure they run against checkpoints without network access (using mocked ingestion).

## Sources

### Primary (HIGH confidence)

- `CLAUDE.md` — Project-wide architecture, data/feature pipeline, checkpointing, and current status (steps 01–07 running, 213 passing tests).
- `.planning/PROJECT.md` — Project purpose, constraints (ETF-only, non-intraday, no auto-trading).
- `.planning/REQUIREMENTS.md` — Formal definitions of DATA-01/02/03 and CONSTR-01/02.
- `.planning/ROADMAP.md` — Mapping of requirements to Phase 1 scope and success criteria.
- `.planning/STATE.md` — Confirms current phase and that v1 implementation is not yet planned at the GSD layer.

### Secondary (MEDIUM confidence)

- Inspection of `ingestion.fred`, `ingestion.multpl`, `ingestion.assets` for concrete ingestion behavior and documentation.

### Tertiary (LOW confidence)

- None used for Phase 1; all key information comes from the repository itself.

## Metadata

**Confidence breakdown:**

| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | Directly derived from `CLAUDE.md`, `pyproject.toml` expectations, and existing `src/market_regime` modules. |
| Architecture | HIGH | Repo layout and pipeline structure are explicitly documented and already implemented end-to-end. |
| Pitfalls | MEDIUM | Based on project docs plus typical time-series/ML issues; should be refined with concrete tests and usage. |

**Research date:** 2026-03-16  
**Valid until:** 2026-04-15 (stable domain; revisit if ingestion/feature stack changes materially)

