#!/usr/bin/env bash
# run_tests.sh — run pytest inside the repo's conda env.
#
# Usage:
#   bash scripts/run_tests.sh
#   bash scripts/run_tests.sh tests/unit/test_returns.py -q
#
# Behavior:
#   - Uses conda env "${TRADING_CRAB_CONDA_ENV:-py310}" by default
#   - Verifies "python -m pytest" works in that env before running tests
#   - Passes all arguments through to pytest

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${TRADING_CRAB_CONDA_ENV:-py310}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Install Miniconda/Conda first, then retry."
  exit 1
fi

if ! conda run -n "${CONDA_ENV_NAME}" python -m pytest --version >/dev/null 2>&1; then
  echo "pytest is not usable in conda env '${CONDA_ENV_NAME}'."
  echo "Try: conda install -n ${CONDA_ENV_NAME} pytest"
  exit 1
fi

cd "${REPO_ROOT}"
exec conda run -n "${CONDA_ENV_NAME}" python -m pytest "$@"

