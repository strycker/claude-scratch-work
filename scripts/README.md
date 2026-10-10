# Scripts

## Platform (active): the weekly product

| Command | What it does |
|---|---|
| `python scripts/build_platform_data.py` | Fetches every source and builds the platform data (`data/checkpoints/platform/`, holdout carve). Needs network and `FRED_API_KEY`. If your network blocks a source (macrotrends 403 or a Yahoo rate limit on a corporate VPN), the build stops and prints the `--allow-missing COL,...` command that builds without it for that run. |
| `python -m trading_crab_lib.platform.report.serving` | Fits the served regime model. It is only needed for `allocation_mode: regime_tilt`. |
| `python -m trading_crab_lib.platform.report.weekly` | Writes the weekly page (`outputs/reports/platform/weekly_report.md`). |
| `python scripts/platform_snapshot.py` | Offline dev snapshots of the platform checkpoints (see `docs/offline_snapshots.md`). |
| `python scripts/recompute_monthly_features.py` | Network-free recompute of `monthly_features` from the cached `monthly_raw`. |
| `python scripts/migrate_*.py` | One-off migrations from 08.1 and 08.3. They are kept for their record tests and retire in 08.5. |

`diagnose_macrotrends*.py` and `diagnose_yahoo_interceptor.sh` help when a source fails to fetch on a new machine.

## Legacy quarterly pipeline (frozen)

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
- Verifies that `trading_crab_lib` can be imported.
- Runs a tiny pytest smoke test (`test_current_regime_models_and_probabilities`).
