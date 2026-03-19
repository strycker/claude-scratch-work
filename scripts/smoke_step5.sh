#!/usr/bin/env bash
# smoke_step5.sh — quick end-to-end check for step 5 (predict) after steps 1–4.
#
# Prerequisites (from a normal pipeline run):
#   - data/processed/features_supervised.parquet  (step 2 causal features)
#   - data/regimes/cluster_labels.parquet         (step 3)
#   - data/raw/macro_raw.parquet OR data/raw/asset_prices.parquet (returns for behavior models)
#
# Usage (from repo root):
#   bash scripts/smoke_step5.sh
#   bash scripts/smoke_step5.sh --verbose
#
# Optional — run full 1→5 in one shot (needs FRED key / network for step 1 if no cache):
#   SMOKE_FULL_PIPELINE=1 bash scripts/smoke_step5.sh
#
# Environment:
#   PYTHON  — interpreter command (default: python3)
#   TRADING_CRAB_CONDA_ENV — conda env name to run in (default: py310)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV_NAME="${TRADING_CRAB_CONDA_ENV:-py310}"
USE_CONDA=0
if command -v conda >/dev/null 2>&1; then
  # If conda is available and the env is usable, prefer it (it should have deps like pyarrow, dotenv, sklearn).
  if conda run -n "${CONDA_ENV_NAME}" "${PYTHON}" --version >/dev/null 2>&1; then
    USE_CONDA=1
  fi
fi

if [[ "${USE_CONDA}" -eq 0 ]]; then
  if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Interpreter not found: ${PYTHON}"
    exit 1
  fi
fi

run_py() {
  if [[ "${USE_CONDA}" -eq 1 ]]; then
    conda run -n "${CONDA_ENV_NAME}" "${PYTHON}" "$@"
  else
    "${PYTHON}" "$@"
  fi
}

FEATURES_SUP="${REPO_ROOT}/data/processed/features_supervised.parquet"
LABELS="${REPO_ROOT}/data/regimes/cluster_labels.parquet"
MACRO="${REPO_ROOT}/data/raw/macro_raw.parquet"
ASSETS="${REPO_ROOT}/data/raw/asset_prices.parquet"

if [[ -n "${SMOKE_FULL_PIPELINE:-}" ]]; then
  echo "smoke_step5: running full pipeline steps 1–5 ..."
  run_py run_pipeline.py --steps 1,2,3,4,5 "$@"
else
  missing=()
  [[ -f "${FEATURES_SUP}" ]] || missing+=("${FEATURES_SUP}")
  [[ -f "${LABELS}" ]] || missing+=("${LABELS}")
  if [[ ! -f "${MACRO}" && ! -f "${ASSETS}" ]]; then
    missing+=("(need ${MACRO} or ${ASSETS} for behavior-model returns)")
  fi
  if (( ${#missing[@]} > 0 )); then
    echo "smoke_step5: missing prerequisites:"
    for m in "${missing[@]}"; do echo "  - ${m}"; done
    echo ""
    echo "Run steps 1–4 first, e.g.:"
  echo "  ${PYTHON} run_pipeline.py --steps 1,2,3,4"
    echo "Or set SMOKE_FULL_PIPELINE=1 to run 1–5 in one invocation."
    exit 1
  fi

  echo "smoke_step5: running step 5 only (steps 1–4 artifacts present) ..."
  run_py run_pipeline.py --steps 5 "$@"
fi

MODEL_DIR="${REPO_ROOT}/outputs/models"
METRICS_DIR="${REPO_ROOT}/outputs/reports/model_metrics"

check_file() {
  local f="$1"
  if [[ ! -f "${f}" ]]; then
    echo "smoke_step5: FAIL — expected file missing: ${f}"
    exit 1
  fi
}

check_file "${MODEL_DIR}/current_regime.pkl"
check_file "${MODEL_DIR}/decision_tree.pkl"
check_file "${MODEL_DIR}/forward_classifiers.pkl"
check_file "${MODEL_DIR}/behavior_models.pkl"
check_file "${METRICS_DIR}/cv_summary.parquet"
check_file "${METRICS_DIR}/per_fold.jsonl"
check_file "${METRICS_DIR}/confusion_matrices.parquet"
check_file "${METRICS_DIR}/calibration.parquet"

echo ""
echo "smoke_step5: OK — step 5 artifacts present under outputs/models/ and outputs/reports/model_metrics/"
