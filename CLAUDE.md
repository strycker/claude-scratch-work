# CLAUDE.md: Trading-Crab project guide

Claude Code reads this file at the start of every session. Keep it short. The long history of the frozen legacy
quarterly pipeline (its ADRs #1–#12, pitfalls P1–P27 and decision log D1–D50) lives verbatim in
`docs/archive/LEGACY-CLAUDE.md`.

## What this is

Trading-Crab produces **weekly, guidance-only portfolio advice** that Glenn executes by hand in Fidelity. It is
long-only, with no options, shorts or crypto. Active development is the **monthly platform**:

- `src/trading_crab_lib/platform/`: the code;
- `config/platform_settings.yaml`: every tuneable value;
- `notebooks/platform/P1–P9`: one notebook per module, each with a human sign-off cell;
- `scripts/build_platform_data.py`: builds the data.

The legacy quarterly pipeline (`src/trading_crab/`, older `src/trading_crab_lib/*.py`, `pipelines/`, `legacy/`)
is a **frozen reference**. Do not develop it.

## Governing principle: KISS (DECISIONS P-07)

The platform must be **human-readable, human-editable and human-testable by one person** with Python scripts and
notebooks.

- Prefer the plainest honest design: fewer modules, fewer config switches, fewer bespoke checks, plain files.
- Spend rigor (mutation proofs, ruling records, high-precision verification layers) only on **decision-bearing
  numbers**.
- When rigor and readability conflict anywhere else, readability wins.

## Where things are decided and tracked

| What | Where |
|---|---|
| Every decision, one row each (edit here first) | `platform_design/DECISIONS.md` |
| Module map M0–M7: interface, files, notebook, gate | `platform_design/MODULE-MAP.md` |
| Build order and lean rules | `REBUILD-FROM-SCRATCH-GUIDE.md` |
| Full design (long-form) | `platform_design/platform_design.md`, ADRs in `platform_design/adr/` |
| Phase status | `.planning/ROADMAP.md`, `.planning/STATE.md` (GSD-managed) |
| Codebase map | `.planning/codebase/*.md` |

**Working mode: lean MVP.**
- One module per phase, at most 3 plans, at most 5 discussion questions.
- Decisions are recorded as DECISIONS rows.
- Glenn creates and merges the PRs.

## How to run

```bash
pip install -e "src/trading_crab_lib/[all,dev]" && pip install -e ".[dev]"   # or: uv sync
cp .env.example .env                       # then add FRED_API_KEY (never commit .env)

python scripts/build_platform_data.py                  # fetch + build platform data (network)
python -m trading_crab_lib.platform.report.weekly      # write the weekly page
python -m trading_crab_lib.platform.report.serving     # fit the served regime model (regime_tilt mode only)
pytest tests/ -q -p no:cacheprovider                   # full suite (offline, ~8 min)
ruff check src/ tests/                                 # lint
jupyter lab notebooks/platform/
```

The weekly page lands in `outputs/reports/platform/weekly_report.md`.

## Honesty rules (do not break)

- **The 2021+ holdout is locked.** Development, tuning and model selection use data through 2020-12 only.
- **The trial registry (`registry/trials.jsonl`) is append-only.** Every evaluated configuration is a row. A
  wiring or verification run uses `--smoke`, which writes no row. Each phase declares its row budget (ADR-0004).
- **Point-in-time data.** Every raw series carries a `publication_lags` entry in config, applied once at ingestion.
  Supervised features are causal only. No look-ahead.
- **P&L uses month-end prices** (E-08). Monthly-average prices are model inputs only.
- **Never re-run the budgeted backtest** (`python -m trading_crab_lib.platform.evaluation.report`) just to
  refresh a page. It rewrites the tracked record.

## Hard rules

- Never commit `.env` or API keys. Secrets come only from environment variables.
- Never modify `legacy/` or the git submodules (`gsd-scratch-work/`, `trading-crab-lib/`, `trading-crab/`).
- **Pickles are an arbitrary-code-execution risk.** No pickle ever arrives through git. Serving models are refit
  locally.
- Live weekly and serving state (executed weights, belief, allocation mode, …) is machine-local and gitignored
  (G-11).
- Data and output files change in git only through a deliberate migration or a budgeted run, never as a side
  effect of a verification run.
- Branches: `claude/<description>`. Never push to `main`. Commits use the conventional format (`feat:`, `fix:`,
  `docs:`, `test:`, `refactor:`, `chore:`).

## Code conventions

- **Python 3.10+.** Every module starts with `from __future__ import annotations`. Use `X | None` and type hints on
  public functions.
- **Library code uses `logging`, never `print`.** `print` is fine in scripts and CLI `main()`.
- **Errors:** catch specific exceptions only, never a bare `except:`. Fail fast on config errors.
- **Files:** `pathlib.Path` for every path. Parquet for DataFrames.
- **Tests:**
  - offline: mock the network, use synthetic fixtures;
  - pandas 2 **and** 3 compatible: use `pct_change(fill_method=None)` everywhere;
  - floats compared at rel 1e-9.
- **Plots:** plotting code lives in `src/trading_crab_lib/platform/plotting/`. Notebooks call it and define no
  inline plotting logic.
- **Lint:** ruff (`E,F,W,I,UP`), line length 127.

## AI model routing (Claude-only)

- **Default session model is Fable.** Keep architecture, code review, behaviour-exercising tests and
  correctness-critical code (look-ahead guards, CV splits, gap-fill math) on it.
- **Delegate down:** well-specified mechanical work, boilerplate and docs go to Sonnet; parallel strong-reasoning
  subtasks go to Opus.
- **No cross-vendor lanes:** the grok and codex implementer agents are disabled.
