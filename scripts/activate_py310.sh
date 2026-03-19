#!/usr/bin/env bash
# activate_py310.sh — source this to activate the repo's conda env.
#
# Usage:
#   source scripts/activate_py310.sh
#
# What it does:
#   1. Locates your conda installation
#   2. Activates the configured env (default: py310)
#   3. Prepends src/ to PYTHONPATH for local imports
#   4. Verifies pytest is importable in the active env

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced, not executed:"
  echo "  source scripts/activate_py310.sh"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${TRADING_CRAB_CONDA_ENV:-py310}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Install Miniconda/Conda first, then retry."
  return 1
fi

CONDA_BASE="$(conda info --base)"
if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Could not locate conda.sh under conda base: ${CONDA_BASE:-<empty>}"
  return 1
fi

# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Activated conda env: ${CONDA_DEFAULT_ENV}"
echo "Python: $(python -c 'import sys; print(sys.executable)')"
python -m pytest --version >/dev/null
echo "pytest check: OK"

