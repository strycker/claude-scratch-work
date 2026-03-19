#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${TRADING_CRAB_CONDA_ENV:-py310}"

echo "[install] Repo root: $REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
  echo "[install] Using conda environment: $ENV_NAME"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.10 -y
  fi
  # shellcheck disable=SC1090
  eval "$(conda shell.bash hook 2>/dev/null || conda shell.zsh hook 2>/dev/null)"
  conda activate "$ENV_NAME"
else
  echo "[install] conda not found; falling back to venv .venv/"
  PYTHON=$(command -v python3 || command -v python || true)
  if [[ -z "$PYTHON" ]]; then
    echo "[install] Python not found, aborting."
    exit 1
  fi
  "$PYTHON" -m venv "$REPO_ROOT/.venv"
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

cd "$REPO_ROOT"

echo "[install] Installing project (dev extras)…"
pip install -e ".[dev]"

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "[install] Copied .env.example → .env (edit FRED_API_KEY)."
fi

if [[ ! -f config/email.local.yaml && -f config/email.example.yaml ]]; then
  cp config/email.example.yaml config/email.local.yaml
  echo "[install] Copied config/email.example.yaml → config/email.local.yaml (edit SMTP creds)."
fi

echo "[install] Running smoke tests…"
pytest tests/test_pipelines_ingest_features.py::test_step01_ingest_writes_macro_raw_without_network \
       tests/test_models_regime.py::test_current_regime_models_and_probabilities -q

echo "[install] Done. Environment is ready."

