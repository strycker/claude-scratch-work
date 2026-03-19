# Scripts

## run_weekly_report.py

Single entry point for the **weekly regime + recommendation report**.

**Usage (from repo root):**

```bash
python scripts/run_weekly_report.py                    # steps 2–7 (cached ingest)
python scripts/run_weekly_report.py --full             # steps 1–7 (full refresh)
python scripts/run_weekly_report.py --plots            # also save figures
python scripts/run_weekly_report.py --verbose          # DEBUG logging
python scripts/run_weekly_report.py --send-email       # also send via SMTP (see config/email.example.yaml)
```

**Outputs:**

- Runs the pipeline (steps 2–7 by default, or 1–7 with `--full`).
- Copies `outputs/reports/weekly_report.md` → `outputs/reports/weekly_YYYY-MM-DD.md`.
- Writes `outputs/reports/email_body.txt` (plain text for email paste or sendmail).
- With `--send-email`, sends the weekly report via SMTP using `config/email.local.yaml` (based on `config/email.example.yaml`).

**Cron example** (e.g. every Monday 9am):

```cron
0 9 * * 1 cd /path/to/gsd-scratch-work && python scripts/run_weekly_report.py
```

Replace `/path/to/gsd-scratch-work` with your repo root. Ensure your environment (venv, `FRED_API_KEY` in `.env` if using `--full`) is active or sourced in the cron job if needed.

## activate_py310.sh

Source this helper to activate the repo's conda env and validate that `pytest`
is usable in the active interpreter.

**Usage:**

```bash
source scripts/activate_py310.sh
```

**What it does:**

- Activates `${TRADING_CRAB_CONDA_ENV:-py310}` using your local conda install.
- Prepends `src/` to `PYTHONPATH` so local imports work in ad hoc shells.
- Runs `python -m pytest --version` as a quick sanity check.

## run_tests.sh

Conda-aware pytest wrapper for the repo.

**Usage:**

```bash
bash scripts/run_tests.sh
bash scripts/run_tests.sh tests/unit/test_returns.py -q
```

**What it does:**

- Runs `python -m pytest` inside `${TRADING_CRAB_CONDA_ENV:-py310}` via `conda run`.
- Fails fast if `pytest` is not importable in that env.
- Passes all extra arguments directly through to pytest.

## smoke_step5.sh

Quick **end-to-end smoke** for **step 5 (predict)** once steps **1–4** have produced the usual artifacts.

**Usage:**

```bash
bash scripts/smoke_step5.sh
bash scripts/smoke_step5.sh --verbose
```

**Prerequisites (default mode — step 5 only):**

- `data/processed/features_supervised.parquet` (step 2)
- `data/regimes/cluster_labels.parquet` (step 3)
- `data/raw/macro_raw.parquet` **or** `data/raw/asset_prices.parquet` (returns for behavior models)

If anything is missing, the script prints what to run (e.g. `python run_pipeline.py --steps 1,2,3,4`).

**Full pipeline mode** (steps 1–5 in one go; step 1 may need `FRED_API_KEY` / network if raw data is not cached):

```bash
SMOKE_FULL_PIPELINE=1 bash scripts/smoke_step5.sh
```

**What it checks after step 5:**

- `outputs/models/current_regime.pkl`, `decision_tree.pkl`, `forward_classifiers.pkl`, `behavior_models.pkl`
- `outputs/reports/model_metrics/cv_summary.parquet`, `per_fold.jsonl`, `confusion_matrices.parquet`, `calibration.parquet`

Uses `PYTHONPATH=src` and `python3` by default; override with `PYTHON=/path/to/python`.

## install_trading_crab.sh

One-shot installer for Trading-Crab on a new machine.

**Usage:**

```bash
bash scripts/install_trading_crab.sh
```

**What it does:**

- Uses conda `${TRADING_CRAB_CONDA_ENV:-py310}` if available, otherwise falls back to `.venv/`.
- Installs the project (including dev extras) via `pip install -e ".[dev]"`.
- Scaffolds `.env` and `config/email.local.yaml` from their example files if missing.
- Runs a small pytest smoke set to catch obvious environment issues early.

## check_env.sh

Quick environment health check.

**Usage:**

```bash
bash scripts/check_env.sh
```

**What it does:**

- Prints the `python` and `pytest` executables and versions in use.
- Verifies that `market_regime` can be imported.
- Runs a tiny pytest smoke test (`test_current_regime_models_and_probabilities`).
