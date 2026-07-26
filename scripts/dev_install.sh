#!/usr/bin/env bash
# dev_install.sh — Create a local .venv and install Trading-Crab for development.
#
# This installs BOTH packages of the monorepo in editable mode with all extras,
# matching the canonical dev environment (trading-crab-lib[all,dev] + trading-crab[dev]).
# After this, `pytest tests/ -q` and the `tradingcrab` CLI both work.
#
# Usage:
#   bash scripts/dev_install.sh              # create .venv (Python 3.10+) and install
#   PYTHON=python3.11 bash scripts/dev_install.sh   # pin the interpreter explicitly
#   bash scripts/dev_install.sh --lite       # skip heavy optional deps (no hdbscan/lightgbm/hmm/statsmodels)
#   bash scripts/dev_install.sh --help
#
# Notes for conda users:
#   `python -m venv` builds the .venv from whatever `python` is active. To base it on
#   your Python 3.11 conda env, activate it first, then run this script:
#       conda activate py311 && bash scripts/dev_install.sh
#   (You do NOT keep the conda env active afterward — activate .venv instead.)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
LITE=false

for arg in "$@"; do
  case "$arg" in
    --lite) LITE=true ;;
    --help|-h)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
step()   { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die()    { printf '\033[0;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ── 1. Pick and verify the interpreter ─────────────────────────────────────────
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
    command -v "$cand" >/dev/null 2>&1 && { PYTHON="$cand"; break; }
  done
fi
[ -n "$PYTHON" ] || die "No python interpreter found. Install Python 3.10+ (or activate your conda env) first."

PYV="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)' \
  || die "Python >= 3.10 required, found $PYV (from '$PYTHON'). Try: PYTHON=python3.11 bash scripts/dev_install.sh"
step "Using $PYTHON (Python $PYV)"

# ── 2. Create the venv ─────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
  yellow "  .venv/ already exists — reusing it (delete it to start fresh)"
else
  step "Creating virtual environment at .venv/"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# Use the venv's own executables directly (no need to `source activate`).
VPY="$VENV_DIR/bin/python"
[ -x "$VPY" ] || die "venv python not found at $VPY (venv creation failed?)"

# ── 3. Upgrade pip tooling ─────────────────────────────────────────────────────
step "Upgrading pip / setuptools / wheel"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel

# ── 4. Editable install: LIBRARY FIRST, then the APP ──────────────────────────
# Order matters: the app depends on `trading-crab-lib>=0.1.2`. Installing the local
# library editable first satisfies that requirement locally so pip does not pull the
# published wheel from PyPI over your working copy.
if [ "$LITE" = true ]; then
  LIB_EXTRAS="[dev]"
  yellow "  --lite: installing library[dev] only (skips hmm/statsmodels/hdbscan/lightgbm/kneed);"
  yellow "          some tests that need those optional deps will be skipped."
else
  LIB_EXTRAS="[all,dev]"
fi

step "Installing library (editable): src/trading_crab_lib/${LIB_EXTRAS}"
"$VPY" -m pip install -e "$REPO_ROOT/src/trading_crab_lib/${LIB_EXTRAS}"

step "Installing app (editable): .[dev]"
"$VPY" -m pip install -e "$REPO_ROOT/.[dev]"

# ── 5. .env scaffold (FRED_API_KEY etc.) ───────────────────────────────────────
if [ ! -f "$REPO_ROOT/.env" ] && [ -f "$REPO_ROOT/.env.example" ]; then
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  yellow "  Created .env from .env.example — add your FRED_API_KEY to it before running the pipeline."
fi

# ── 6. Smoke check ─────────────────────────────────────────────────────────────
step "Verifying the install"
"$VPY" - <<'PY'
import numpy, pandas, sklearn, scipy   # core runtime
import trading_crab_lib                 # library package importable
import trading_crab                     # app package importable
print(f"  numpy {numpy.__version__} | pandas {pandas.__version__} | "
      f"scikit-learn {sklearn.__version__} | scipy {scipy.__version__}")
print("  trading_crab_lib + trading_crab import OK")
PY

green ""
green "Done. Next steps:"
green "  source .venv/bin/activate"
green "  pytest tests/ -q"
green ""
green "  (Deactivate with: deactivate)"
