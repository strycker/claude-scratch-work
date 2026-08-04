# MIGRATION-PLAN.md — Migrating the L0–L4 Platform to `strycker/trading-crab`

**Purpose:** Move the verified regime-conditional platform out of the AI sandbox
(`strycker/claude-scratch-work`) and into the public two-package repo
(`strycker/trading-crab`), where development continues.

**Target repo:** `https://github.com/strycker/trading-crab`
**Structure:** Two-package layout (`trading-crab` app + `trading-crab-lib` library).
**Requirement:** MIG-01 · **GSD Phase:** 7 (Migration to Public Repo)

---

> ## ⚠️ This plan replaces the 2026-04 version
>
> The previous MIGRATION-PLAN.md (Q0–Q9) migrated the **legacy quarterly 9-step
> pipeline** — ingest → features → clustering → regime → predict → assets → diagnostics
> → tactics → CLI. It was written in April 2026, before the platform existed.
>
> That is **not** what we are migrating. `.planning/ROADMAP.md` Phase 7 targets the
> **L0–L4 platform** built and verified across GSD Phases 1–5. Following the old plan
> would port 10,935 lines of superseded quarterly code and none of the 6,977 lines that
> five phases of verification actually cover.
>
> The old plan is archived at `docs/archive/MIGRATION-PLAN-quarterly-2026-04.md`.
> The legacy quarterly pipeline **stays in `claude-scratch-work`** as reference, in the
> same spirit as `legacy/unified_script.py`. See §7.

---

## 1. What Migrates

| Migrates | Does NOT migrate |
|---|---|
| `src/trading_crab_lib/platform/` (6,977 LOC, 48 modules) | Legacy lib modules (10,935 LOC) |
| 31 `tests/unit/test_platform_*.py` (378 tests) | 35 legacy test files |
| `tests/integration/test_mini_backtest.py` | `tests/integration/test_mini_pipeline.py` |
| `config/platform_settings.yaml` (18 sections) | `config/settings.yaml` (quarterly) |
| `config/accounts/example.yaml` | `config/regime_labels.yaml` |
| Platform notebooks `P1`–`P6` (**built in Phase 6**) | `notebooks/01–12` (quarterly) |
| `docs/splicing_rules.md`, `vintage_alignment.md`, `paid_provider_seams.md` | `docs/archive/` |
| `registry/trials.jsonl` (the honesty ledger — git-tracked by design) | `pipelines/01–09`, `run_pipeline.py` |
| `scripts/build_platform_data.py`, `dev_install.sh` | `scripts/evaluate_{divergence,momentum}.py` |
| `platform_design/platform_design.md` | `legacy/`, `ideas/`, `gsd-scratch-work/` |

---

## 2. The Coupling Surface — why this is cheap

The dependency direction is clean and one-way. Verified by import trace:

```
platform/  →  legacy lib :  4 narrow seams
legacy lib →  platform/  :  ZERO
```

Everything `platform/` borrows from the legacy library:

| # | Seam | Imported by | Resolution |
|---|---|---|---|
| **M1** | `checkpoints.CheckpointManager` | 5 platform modules | Vendor the class into `platform/checkpoints.py` (already wraps it) |
| **M2** | `ingestion.multpl._scrape_raw_rows`, `_SUFFIX_MAP`, `RATE_LIMIT_SECONDS` | `platform/ingestion/macro_monthly.py` | Vendor 3 helpers into `platform/ingestion/_scrapers.py` |
| **M3** | `ingestion.macrotrends._extract_json_data`, `HEADERS` | `platform/ingestion/macro_monthly.py` | Vendor 2 helpers into the same module |
| **M4** | `email.{load_email_config, build_weekly_email_body, send_weekly_email}` | `platform/report/weekly.py` | Vendor into `platform/report/email.py` |

**That is the entire coupling surface.** Cutting these four seams (step **P0**) makes
`platform/` a standalone package, after which migration is a directory move.

---

## 3. Prerequisites — do NOT start before these

| # | Prerequisite | Why it blocks |
|---|---|---|
| **PRE-1** | GSD **Phase 6 (Platform Notebook Suite)** complete | Every step below validates via "run the notebook." Today `platform/` has **zero** notebooks — all 12 cover the quarterly pipeline. Without them there is nothing to validate against. |
| **PRE-2** | Phase 5 closed | ✅ Done 2026-08-04 (`05-VERIFICATION.md` Phase Closure Record) |
| **PRE-3** | Full suite green on `main` | ✅ 1157 passing |
| **PRE-4** | Target repo `trading-crab` reachable and writable | Currently an empty gitlink submodule here |

---

## 4. Migration Steps (P0 → P6)

Work P-steps in order. After each: run the listed tests **in the target repo**, open the
paired notebook, verify manually, then commit. Never skip ahead.

### P0 — Decouple: cut the four seams *(done HERE, before any move)*

Runs in `claude-scratch-work`, not the target. Ends with `platform/` importing nothing
from the legacy library.

| Tag | Description | Files |
|---|---|---|
| P0.1 | Vendor `CheckpointManager` (M1) | `platform/checkpoints.py` |
| P0.2 | Vendor multpl + macrotrends scraper helpers (M2, M3) | `platform/ingestion/_scrapers.py`, `macro_monthly.py` |
| P0.3 | Vendor email helpers (M4) | `platform/report/email.py`, `report/weekly.py` |
| P0.4 | Add an import-guard test asserting zero non-platform `trading_crab_lib` imports | `tests/unit/test_platform_standalone.py` |

**Exit:** `grep -r "from trading_crab_lib\." src/trading_crab_lib/platform | grep -v platform` returns nothing, and the guard test enforces it. Full suite still green.

### P1 — Target repo skeleton

| Tag | Description | Files |
|---|---|---|
| P1.1 | Two-package layout, pyprojects, `py.typed` | `pyproject.toml`, `src/trading_crab_lib/pyproject.toml` |
| P1.2 | Dev install script + requirements | `scripts/dev_install.sh`, `requirements*.txt` |
| P1.3 | Test scaffold (`conftest.py`, isolation fixture) | `tests/conftest.py`, `tests/{unit,integration}/__init__.py` |
| P1.4 | `.gitignore`, `.env.example`, `Makefile` — **`registry/` must stay trackable** | root |

**Validate:** `pip install -e "src/trading_crab_lib/[all,dev]" && pip install -e ".[dev]"`; `pytest tests/ -q` collects 0 with no errors.

### P2 — L0: data spine

| Tag | Description | Files |
|---|---|---|
| P2.1 | `platform_settings.yaml` — `data`, `fred_monthly`, `multpl_monthly`, `macrotrends_monthly`, `universe`, `splice`, `taxonomy` | `config/platform_settings.yaml` |
| P2.2 | Config loader + checkpoint namespace | `platform/config.py`, `platform/checkpoints.py`, `platform/__init__.py` |
| P2.3 | Taxonomy (fast/slow/agency + lean set) | `platform/taxonomy.py` |
| P2.4 | Splicing engine + long-history synthetics | `platform/splice.py`, `docs/splicing_rules.md` |
| P2.5 | Ingestion: monthly macro, daily macro, daily prices, ALFRED, provider stubs | `platform/ingestion/*.py`, `docs/vintage_alignment.md`, `docs/paid_provider_seams.md` |
| P2.6 | Monthly transforms + spine builder | `platform/transforms_monthly.py` |
| P2.7 | Data build script | `scripts/build_platform_data.py` |
| P2.8 | Tests | `test_platform_{splice,alfred,macro_ingest,macro_daily,prices_ingest,taxonomy,transforms,paid_provider_stubs}.py` |

**Validate:** those 8 test files pass; `python scripts/build_platform_data.py` with a live `FRED_API_KEY` produces a ~1962-start `monthly_features` checkpoint; **notebook `P1_data_spine.ipynb` runs clean.**

### P3 — Honesty rails *(before any model — non-negotiable ordering)*

| Tag | Description | Files |
|---|---|---|
| P3.1 | Config: `holdout`, `registry`, `cv` | `config/platform_settings.yaml` |
| P3.2 | Physical 2021+ holdout carve | `platform/honesty/holdout.py` |
| P3.3 | Append-only JSONL trial registry | `platform/honesty/registry.py`, `registry/trials.jsonl` |
| P3.4 | `PurgedEmbargoedKFold` | `platform/honesty/cv.py` |
| P3.5 | Causal gating + gap/detection-lag metrics | `platform/honesty/{gating,gap_lag}.py` |
| P3.6 | Walk-forward runner | `platform/honesty/walkforward.py` |
| P3.7 | Tests | `test_platform_{holdout,registry,cv,gating,gap_lag,walkforward}.py` |

**Validate:** those 6 files pass; the default checkpoint manager provably cannot read a
2021+ row; `registry/trials.jsonl` is **not** gitignored (`git check-ignore` returns
nothing — there is a test for this).

> Design §14 requires the honesty framework installed **before the first model is
> tuned.** Preserve this ordering in the target repo — it is the product, not overhead.

### P4 — L1/L2: regime labeling & prediction

| Tag | Description | Files |
|---|---|---|
| P4.1 | Config: `labeling` (K=5, λ=52, restarts=10, embargo=12) | `config/platform_settings.yaml` |
| P4.2 | Jump model (exact DP decode, multi-restart, canonicalization) | `platform/labeling/jump_model.py` |
| P4.3 | Labeling diagnostics, persistence, label churn | `platform/labeling/diagnostics.py` |
| P4.4 | Calibrated nowcaster | `platform/prediction/nowcaster.py` |
| P4.5 | Empirical transition matrix | `platform/prediction/transition_matrix.py` |
| P4.6 | Tests | `test_platform_{labeling,nowcaster,transition_matrix}.py` |

**Validate:** those 3 files pass; **`P3_regime_labeling.ipynb` and `P4_nowcaster.ipynb` run clean** — this is the human-in-the-loop gate where the 5 regimes are checked against real economic history.

### P5 — L3/L4: assets, allocation, report, tripwire

| Tag | Description | Files |
|---|---|---|
| P5.1 | Config: `allocation`, `report`, `tripwire` | `config/platform_settings.yaml` |
| P5.2 | Returns-by-regime + EWMA vol | `platform/assets/{returns,vol}.py` |
| P5.3 | Vol-targeted tilt + hysteresis | `platform/allocation/{tilt,hysteresis}.py` |
| P5.4 | Weekly report + holdings (+ vendored email, M4) | `platform/report/*.py`, `config/accounts/example.yaml` |
| P5.5 | Daily tripwire | `platform/tripwire/monitor.py` |
| P5.6 | Tests | `test_platform_{returns,vol,tilt,hysteresis,report_weekly,holdings,tripwire}.py` |

**Validate:** those 7 files pass; weekly report renders to markdown; tripwire CLI exits 0 and prints one escalation value; **`P5_assets_allocation.ipynb` runs clean.**

### P6 — Backtest & evaluation, CI, docs

| Tag | Description | Files |
|---|---|---|
| P6.1 | Config: `backtest` (incl. `feature_min_history`, `nowcaster_cv_splits`) | `config/platform_settings.yaml` |
| P6.2 | Costs/turnover + walk-forward driver | `platform/backtest/{costs,driver}.py` |
| P6.3 | Baseline gauntlet + no-regime ablation | `platform/backtest/baselines.py` |
| P6.4 | KPIs, sojourn/lag headline, model metrics, report assembly | `platform/evaluation/*.py` |
| P6.5 | Integration test | `tests/integration/test_mini_backtest.py` |
| P6.6 | Tests | `test_platform_backtest_*.py`, `test_platform_evaluation_*.py` |
| P6.7 | CI, pre-commit, ruff/mypy | `.github/workflows/`, `.pre-commit-config.yaml` |
| P6.8 | Platform-first `README.md` + `CLAUDE.md`; `platform_design.md` | root |

**Validate:** **full suite green in target-repo CI** (not just locally); a real
1972–2020 run reproduces the numbers below; **`P6_backtest_evaluation.ipynb` runs clean.**

---

## 5. Acceptance — the migration is done when

1. `pytest tests/ -q` is green in the target repo **and in its CI**, ≥ 378 platform tests.
2. `platform/` imports nothing from the legacy library (guard test P0.4 enforces).
3. A real 1972–2020 walk-forward reproduces the reference numbers (§6) within tolerance.
4. All six notebooks `P1`–`P6` run top-to-bottom without error.
5. `registry/trials.jsonl` is git-tracked and gains exactly 2 rows per full run.
6. No DSR computed; no 2021+ read on any dev path.
7. README/docs describe the regime-conditional platform, not the quarterly pipeline.

## 6. Reference numbers (post-activation real run, 2026-08)

The migrated code must reproduce these. Divergence means something broke in transit.

| Metric | Value |
|---|---|
| Walk-forward window | 1972-01 → 2020-12 (588 steps, 512 predicted) |
| Terminal log wealth (strategy) | 111.06 |
| Max drawdown | −66.2% (12 months underwater) |
| No-regime ablation | 113.89 (delta **−2.83** — regime layer does not pay rent yet) |
| Baselines: SPY / 60-40 / Faber | 5.68 / 131.03 / 1.14 |
| Median sojourn / lag → ratio | 84.5 / 161.5 → 0.52 *(2 of 6 transitions resolved)* |
| Multiclass Brier | 0.20 |
| Registry rows per run | 2 (strategy + ablation) |

**These are honest, not good.** The regime layer underperforming its own ablation is the
expected design §14 Phase-1 result ("beats nothing yet — that's fine"). Migrating a
result you can trust matters more than migrating a flattering one. Do not tune to improve
these during migration — that is the next milestone's work, and tuning here would burn
trials against the registry for no gain.

## 7. What stays behind, and why

The **legacy quarterly pipeline** (`pipelines/01–09`, `notebooks/01–12`, non-`platform/`
library modules, `run_pipeline.py`, the 9-step orchestrator in `src/trading_crab/`) stays
in `claude-scratch-work`.

Reasons:

1. **Superseded.** The platform replaces every layer it provided.
2. **Not a baseline.** Phase 5's gauntlet is SPY / 60-40 / Faber — the quarterly pipeline
   is not measured against anything.
3. **Parameter bloat.** 101 `clustering_features` → PCA(5), versus the platform's 13
   curated named features. Carrying it forward re-imports the complexity we removed.
4. **Nearly zero reuse.** Four small seams, all vendored in P0.

This is simplification by *not carrying forward* — no deletion, no risk. The code keeps
working here for reference.

## 8. Delivery convention

Ask **"Show me P4.2"** to get exact file contents for that step: full contents for new
files, diffs for modified ones, and the pytest command that validates it.

## 9. Dependencies

```
PRE-1 (Phase 6 notebooks)
        ↓
P0 decouple  →  P1 skeleton  →  P2 L0 data
                                     ↓
                              P3 honesty rails   ← must precede any model
                                     ↓
                              P4 L1/L2 regime
                                     ↓
                              P5 L3/L4 alloc
                                     ↓
                              P6 backtest + CI + docs
```

Do not start P*n+1* until P*n* is human-validated in the target repo.
