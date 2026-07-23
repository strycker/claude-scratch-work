#!/bin/bash
# SessionStart hook — provision a complete Python dev environment for
# trading-crab inside Claude Code on the web.
#
# Each web session runs in a fresh, ephemeral container with the repo cloned
# but nothing installed. This script builds a .venv with BOTH packages
# (library + app) editable-installed plus every optional extra, so tests,
# linters, and the pipeline run out of the box — a zero-skip `pytest tests/`.
#
# It also appends the venv to the session PATH via $CLAUDE_ENV_FILE so every
# shell Claude opens uses the venv interpreter without manual activation.
#
# Properties: remote-only, idempotent (safe to re-run), non-interactive.
set -euo pipefail

# Web/remote sessions only — local machines keep their own environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"

VENV=".venv"
PYBIN="$VENV/bin/python"
LOG="$(mktemp -t tc-session-start.XXXXXX.log)"

# Editable installs: library with all extras (ingestion, plotting, hmm,
# clustering-extras, boosting) + dev tools, then the app with dev tools.
# The lib's [all,dev] covers every optional dependency needed for a zero-skip
# test run (hmmlearn, statsmodels, hdbscan, lightgbm, cssselect, kneed, ...).
LIB_TARGET="src/trading_crab_lib/[all,dev]"
APP_TARGET=".[dev]"

install_env() {
  if command -v uv >/dev/null 2>&1; then
    # uv: prebuilt wheels for scipy/sklearn/hdbscan/lightgbm — seconds, not minutes.
    [ -x "$PYBIN" ] || uv venv "$VENV"
    uv pip install --python "$PYBIN" -e "$LIB_TARGET" -e "$APP_TARGET"
  else
    # Fallback for base images without uv.
    [ -x "$PYBIN" ] || python3 -m venv "$VENV"
    "$PYBIN" -m pip install --upgrade pip
    "$PYBIN" -m pip install -e "$LIB_TARGET" -e "$APP_TARGET"
  fi
}

if ! install_env >"$LOG" 2>&1; then
  echo "session-start: dependency install FAILED — last 40 lines:" >&2
  tail -40 "$LOG" >&2
  exit 1
fi
rm -f "$LOG"

# Make the venv the default interpreter for every shell in this session.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export VIRTUAL_ENV=\"$PWD/$VENV\""
    echo "export PATH=\"$PWD/$VENV/bin:\$PATH\""
  } >> "$CLAUDE_ENV_FILE"
fi

echo "session-start: .venv ready (trading-crab-lib[all,dev] + trading-crab[dev])"
