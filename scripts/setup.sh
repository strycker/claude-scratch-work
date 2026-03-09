#!/usr/bin/env bash
# setup.sh — One-shot environment setup for Trading-Crab.
#
# Usage:
#   bash scripts/setup.sh           # standard install
#   bash scripts/setup.sh --dev     # include testing + JupyterLab extras
#   bash scripts/setup.sh --help
#
# What this script does:
#   1. Verifies Python >= 3.10
#   2. Creates a virtual environment at .venv/ (skipped if already present)
#   3. Installs pinned dependencies from requirements.txt (or requirements-dev.txt)
#   4. Optionally installs k-means-constrained (for balanced clustering)
#   5. Copies .env.example → .env if .env is missing
#   6. Creates runtime output directories (data/, outputs/)
#   7. Prints a quick-start reminder

set -euo pipefail

# ── helpers ────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
die()    { red "ERROR: $*"; exit 1; }
step()   { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ── argument parsing ───────────────────────────────────────────────────────────

DEV_MODE=false
for arg in "$@"; do
  case "$arg" in
    --dev)   DEV_MODE=true ;;
    --help|-h)
      echo "Usage: bash scripts/setup.sh [--dev]"
      echo ""
      echo "  --dev   Install testing + JupyterLab extras (requirements-dev.txt)"
      echo ""
      exit 0
      ;;
    *) die "Unknown argument: $arg" ;;
  esac
done

# ── 1. Python version check ────────────────────────────────────────────────────

step "Checking Python version"

PYTHON=$(command -v python3 || command -v python || true)
[[ -z "$PYTHON" ]] && die "Python not found. Install Python 3.10+ and try again."

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]]; then
  die "Python 3.10+ required, found $PY_VERSION. Install a newer Python and retry."
fi
green "  Python $PY_VERSION — OK"

# ── 2. Virtual environment ─────────────────────────────────────────────────────

step "Setting up virtual environment at .venv/"

if [[ -d "$VENV_DIR" ]]; then
  yellow "  .venv/ already exists — skipping creation"
else
  "$PYTHON" -m venv "$VENV_DIR"
  green "  Created .venv/"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"

# Upgrade pip quietly
"$PIP" install --quiet --upgrade pip

# ── 3. Install dependencies ────────────────────────────────────────────────────

if [[ "$DEV_MODE" == true ]]; then
  step "Installing dev dependencies (requirements-dev.txt)"
  "$PIP" install --quiet -r "$REPO_ROOT/requirements-dev.txt"
  green "  Runtime + dev extras installed"
else
  step "Installing runtime dependencies (requirements.txt)"
  "$PIP" install --quiet -r "$REPO_ROOT/requirements.txt"
  green "  Runtime dependencies installed"
fi

# ── 4. Optional: k-means-constrained ──────────────────────────────────────────

step "Checking for optional k-means-constrained"

if "$VENV_DIR/bin/python" -c "import k_means_constrained" 2>/dev/null; then
  green "  k-means-constrained already installed"
else
  echo "  k-means-constrained is optional but recommended for balanced clustering."
  read -r -p "  Install it now? [y/N] " ans
  case "$ans" in
    [yY]*) "$PIP" install --quiet "k-means-constrained>=0.7"; green "  k-means-constrained installed" ;;
    *)     yellow "  Skipped — pipeline will fall back to plain KMeans (balanced_cluster column still produced)" ;;
  esac
fi

# ── 5. .env file ──────────────────────────────────────────────────────────────

step "Setting up .env"

ENV_FILE="$REPO_ROOT/.env"
ENV_EXAMPLE="$REPO_ROOT/.env.example"

if [[ -f "$ENV_FILE" ]]; then
  yellow "  .env already exists — skipping"
else
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  green "  Copied .env.example → .env"
  yellow "  ACTION REQUIRED: edit .env and set FRED_API_KEY"
  yellow "    Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
fi

# ── 6. Runtime directories ────────────────────────────────────────────────────

step "Creating runtime directories"

for dir in \
  "$REPO_ROOT/data/raw" \
  "$REPO_ROOT/data/processed" \
  "$REPO_ROOT/data/regimes" \
  "$REPO_ROOT/data/checkpoints" \
  "$REPO_ROOT/outputs/plots" \
  "$REPO_ROOT/outputs/models" \
  "$REPO_ROOT/outputs/reports"
do
  mkdir -p "$dir"
done
green "  data/ and outputs/ structure created"

# ── 7. Quick-start reminder ────────────────────────────────────────────────────

printf '\n'
green "══════════════════════════════════════════════════"
green "  Setup complete!"
green "══════════════════════════════════════════════════"
printf '\n'
echo "Next steps:"
echo ""
echo "  1. Activate the virtual environment:"
echo "       source .venv/bin/activate"
echo ""
if ! grep -q "^FRED_API_KEY=[^y]" "$ENV_FILE" 2>/dev/null; then
  echo "  2. Add your FRED API key to .env:"
  echo "       FRED_API_KEY=your_key_here"
  echo "     (free key: https://fred.stlouisfed.org/docs/api/api_key.html)"
  echo ""
  echo "  3. Run the full pipeline:"
else
  echo "  2. Run the full pipeline:"
fi
echo "       python run_pipeline.py --refresh --recompute --plots --market-code grok"
echo ""
echo "  Or run individual steps:"
echo "       python run_pipeline.py --steps 3,4,5,6,7 --plots --market-code grok"
echo ""
if [[ "$DEV_MODE" == true ]]; then
  echo "  Run tests:"
  echo "       pytest tests/ -v"
  echo ""
  echo "  Launch notebooks:"
  echo "       jupyter lab notebooks/"
  echo ""
fi
