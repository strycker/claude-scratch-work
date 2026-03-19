#!/usr/bin/env bash
# clean_workspace.sh — remove runtime data & outputs, keep cold-start inputs
#
# Keeps:
#   - Seed files directly under ./data (e.g. grok_*.pickle, .xlsx, CSVs)
# Removes:
#   - data/checkpoints/, data/raw/, data/processed/, data/regimes/
#   - outputs/ (plots, models, reports)
#   - .pytest_cache/, __pycache__/, and common build artifacts
#
# Usage (from repo root):
#   bash scripts/clean_workspace.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[clean] Repo root: $REPO_ROOT"

# Sanity check to avoid accidental use outside this repo
if [[ ! -f "run_pipeline.py" ]]; then
  echo "[clean] run_pipeline.py not found; aborting (not at repo root?)."
  exit 1
fi

echo "[clean] Removing data subdirectories (keeping seed files in ./data)…"

# Remove data subdirs but keep files directly under ./data (e.g. grok_*.pickle, .xlsx)
rm -rf data/checkpoints data/raw data/processed data/regimes

# Recreate expected data layout
mkdir -p data/raw data/processed data/regimes data/checkpoints

echo "[clean] Removing outputs (plots, models, reports)…"
rm -rf outputs/*
mkdir -p outputs/plots outputs/models outputs/reports

echo "[clean] Removing Python/pytest caches…"
rm -rf .pytest_cache
find "$REPO_ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} +

echo "[clean] Removing build artifacts…"
rm -rf build dist *.egg-info

echo "[clean] final cleansing... removing remaining pycache files, checkpoints, egg-info stuff, etc."
rm -rf .pytest_cache
find . -type f -name "*.pyc" -delete
find . -type f -name "*__pycache__*" -delete
find . -type d -name "*__pycache__*" -delete
find . -type f -name "*checkpoint.ipynb*" -delete
find . -type d -name "*.ipynb_checkpoints*" -delete
find . -type f -name "*egg-info*" -delete
find . -type d -name "*egg-info*" -delete
find . -type f -name "*.egg-info*" -delete
find . -type d -name "*.egg-info*" -delete
find . -type d -name "*.egg-info*" -exec rm -rf {} \;

echo "[clean] Done. Seed files directly under ./data were preserved."

