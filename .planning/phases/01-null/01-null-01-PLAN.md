---
phase: 01-null
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - config/settings.yaml
  - run_pipeline.py
  - pipelines/01_ingest.py
  - src/market_regime/config.py
  - src/market_regime/runtime.py
  - src/market_regime/io/checkpoints.py
  - src/market_regime/ingestion/fred.py
  - src/market_regime/ingestion/multpl.py
  - src/market_regime/ingestion/assets.py
autonomous: true
requirements:
  - DATA-01
  - DATA-02
  - CONSTR-01
  - CONSTR-02
must_haves:
  truths:
    - "The pipeline can ingest macro series and ETF prices for the configured ETF universe over the intended historical window without manual intervention."
    - "Typical runs can rely on parquet checkpoints instead of re-scraping every time while still producing correct downstream artifacts."
    - "Ingestion and checkpoint behavior for macro data and ETF prices is observable via logs and a single CLI entrypoint."
    - "All ingested assets respect the ETF-only, non-intraday, non-auto-trading constraints."
  artifacts:
    - path: "config/settings.yaml"
      provides: "Single source of truth for macro series, ETF universe, and data cadence configuration used by ingestion."
    - path: "pipelines/01_ingest.py"
      provides: "Step-01 ingestion orchestrator using RunConfig and CheckpointManager for macro_raw and asset_prices."
    - path: "run_pipeline.py"
      provides: "CLI entrypoint wiring RunConfig flags to Step 01 ingestion behavior."
    - path: "src/market_regime/io/checkpoints.py"
      provides: "CheckpointManager implementation for parquet checkpoints and freshness checks."
    - path: "src/market_regime/ingestion/fred.py"
      provides: "Config-driven FRED macro ingestion with publication-lag shifts and quarterly resampling."
    - path: "src/market_regime/ingestion/multpl.py"
      provides: "Config-driven multpl.com macro ingestion with correct units and rate limiting."
    - path: "src/market_regime/ingestion/assets.py"
      provides: "ETF price ingestion for configured tickers with fallback chain and quarterly resampling."
  key_links:
    - from: "pipelines/01_ingest.py"
      to: "src/market_regime/ingestion.fred.fetch_all"
      via: "function call using cfg from market_regime.config.load()"
    - from: "pipelines/01_ingest.py"
      to: "src/market_regime/ingestion.multpl.fetch_all"
      via: "function call using cfg from market_regime.config.load()"
    - from: "pipelines/01_ingest.py"
      to: "src/market_regime/ingestion.assets.fetch_all"
      via: "function call using cfg and ETF universe from settings.yaml"
    - from: "pipelines/01_ingest.py"
      to: "src/market_regime/io.checkpoints.CheckpointManager"
      via: "save_parquet and is_fresh calls for macro_raw and asset_prices checkpoints"
    - from: "run_pipeline.py"
      to: "pipelines/01_ingest.main"
      via: "steps/flags dispatch controlled by RunConfig"
---

<objective>
Establish a config-driven, checkpointed ingestion step that pulls macro series and ETF prices for the configured ETF universe and exposes it through a single CLI entrypoint, while enforcing ETF-only and non-intraday constraints at the ingestion layer.

Purpose: Turn the existing ingestion modules and CheckpointManager into a stable, documented Step 01 contract that satisfies DATA-01/02 and supports CONSTR-01/02.
Output: Updated config, ingestion pipeline code, and CLI wiring for macro_raw and asset_prices checkpoints.
</objective>

<execution_context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/01-null/01-RESEARCH.md
</execution_context>

<context>
@CLAUDE.md
@config/settings.yaml
@pipelines/01_ingest.py
@run_pipeline.py
@src/market_regime/config.py
@src/market_regime/runtime.py
@src/market_regime/io/checkpoints.py
@src/market_regime/ingestion/fred.py
@src/market_regime/ingestion/multpl.py
@src/market_regime/ingestion/assets.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Align ingestion step with config-driven ETF universe and macro series</name>
  <files>
config/settings.yaml
src/market_regime/config.py
src/market_regime/ingestion/fred.py
src/market_regime/ingestion/multpl.py
src/market_regime/ingestion/assets.py
  </files>
  <action>
- Ensure `config/settings.yaml` cleanly defines:
  - Macro series (FRED and multpl) under their documented config keys, including `shift` flags and quarterly cadence.
  - The ETF universe (tickers only) under a single canonical section (e.g. `assets.etfs`) that ingestion code uses exclusively.
- In `src/market_regime/config.py`, verify or add helpers so ingestion code can obtain the ETF universe and macro config solely from `load()`, without hard-coded tickers or URLs.
- In `src/market_regime/ingestion/fred.py` and `multpl.py`, confirm ingestion uses the config-driven series lists, preserves publication-lag shifts, and enforces quarterly resolution.
- In `src/market_regime/ingestion/assets.py`, ensure ETF tickers are drawn only from the config ETF universe, prices are resampled to monthly/quarterly (no intraday), and no non-ETF tickers or daily-series shortcuts are allowed.
- Add or tighten logging so that macro series, ETF tickers, and date ranges used during ingestion are clearly visible in logs to support Phase 1 validation.
  </action>
  <verify>
- Run `pytest tests/ -k "ingestion" -v` to confirm ingestion tests still pass and reflect the config-driven ETF universe and macro series.
- Optionally run `python pipelines/01_ingest.py` (with appropriate `.env` for FRED) to confirm macro_raw and asset_prices are produced without hard-coded tickers or manual tweaks.
  </verify>
  <done>
- Macro and ETF ingestion functions load all tickers/series solely from `config/settings.yaml` and respect ETF-only, non-intraday constraints.
- Logs clearly show which macro series and ETF tickers were ingested and over what date range.
  </done>
</task>

<task type="auto">
  <name>Task 2: Standardize checkpointed ingestion orchestration and CLI wiring</name>
  <files>
pipelines/01_ingest.py
run_pipeline.py
src/market_regime/runtime.py
src/market_regime/io/checkpoints.py
  </files>
  <action>
- In `src/market_regime/runtime.py`, verify `RunConfig` exposes flags for refreshing source datasets and recomputing derived datasets consistent with `CLAUDE.md`.
- In `pipelines/01_ingest.py`, ensure a `main(run_cfg: RunConfig)` function:
  - Loads config via `market_regime.config.load()`.
  - Instantiates a `CheckpointManager`.
  - Uses `run_cfg.refresh_source_datasets` to decide when to bypass `is_fresh` and force re-scrapes.
  - Writes `macro_raw` and `asset_prices` checkpoints via `save_parquet` under consistent names expected by later steps.
- Make sure `pipelines/01_ingest.py` uses only quarterly (or monthly→quarterly) data and never introduces sub-daily resolutions.
- In `run_pipeline.py`, confirm that:
  - Step selection and flags route to `pipelines/01_ingest.main(run_cfg)` for Step 1.
  - `--refresh` and `--recompute` semantics for Step 1 match the behavior documented in `CLAUDE.md` and Phase 1 RESEARCH.
- If needed, add minimal CLI help text or comments in `run_pipeline.py` to document how ingestion and checkpoints behave under different flag combinations (refresh vs reuse).
  </action>
  <verify>
- Run `python pipelines/01_ingest.py` to ensure it completes without errors and produces `macro_raw` and `asset_prices` checkpoints under `data/checkpoints/`.
- Run `python run_pipeline.py --steps 1` to confirm Step 1 executes via the unified CLI with the same behavior.
- Re-run `python run_pipeline.py --steps 1` without `--refresh` and verify via logs and elapsed time that checkpoints are reused rather than re-scraped (when fresh).
  </verify>
  <done>
- A single CLI entrypoint (`run_pipeline.py` and `pipelines/01_ingest.py`) orchestrates ingestion using `RunConfig` and `CheckpointManager`.
- `macro_raw` and `asset_prices` checkpoints are produced and reused according to freshness and `--refresh`/`--recompute` flags.
- No ingestion path operates on non-ETF assets or intraday data.
  </done>
</task>

</tasks>

<verification>
- `pytest tests/ -k "ingestion or checkpoint" -v` passes, covering macro + ETF ingestion and checkpointed pipeline behavior.
- `python pipelines/01_ingest.py` and `python run_pipeline.py --steps 1` both succeed and write `macro_raw` and `asset_prices` checkpoints to `data/checkpoints/`.
- Logs for a typical Step 1 run show the configured ETF universe, macro series, and date ranges with no non-ETF assets or sub-daily frequencies.
</verification>

<success_criteria>
- Ingestion for macro series and the ETF universe runs end-to-end via the standard CLI without manual intervention.
- Checkpoints for macro_raw and asset_prices are created and reused as intended, making re-scrapes optional for typical workflows.
- All ingestion paths respect ETF-only, non-intraday constraints, and this behavior is visible in configuration and logs.
</success_criteria>

<output>
After completion, ensure `.planning/ROADMAP.md` Phase 1 reflects this plan under **Plans** and that ingestion/checkpoint behavior is referenced in any future Phase 1 verification or UAT documents.
</output>

