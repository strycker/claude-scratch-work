#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[check_env] python: $(command -v python || true)"
python --version || echo "[check_env] python not found"

echo "[check_env] pytest: $(command -v pytest || true)"
pytest --version || echo "[check_env] pytest not found"

echo "[check_env] Importing market_regime…"
python - <<'PY'
import sys
try:
    import market_regime  # noqa: F401
    print("[check_env] Imported market_regime OK")
except Exception as e:
    print("[check_env] FAILED to import market_regime:", e)
    sys.exit(1)
PY

echo "[check_env] Running tiny smoke test…"
pytest tests/test_models_regime.py::test_current_regime_models_and_probabilities -q

echo "[check_env] Environment looks healthy."

