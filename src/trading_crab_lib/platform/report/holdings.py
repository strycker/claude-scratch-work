"""
Per-account holdings YAML loader (L4-03, design §7, 04-CONTEXT.md D-01) —
warn-don't-fail, cash-aware, no-silent-normalize.

Deliberately NOT the incumbent ``trading_crab_lib.config.load_portfolio()``:
that loader silently normalizes weights to sum=1 and has no cash concept,
which conflicts with D-01's explicit "validate with tolerance, warn don't
fail" requirement and the cash-sleeve framing (FZFXX/SPAXX). This module is a
new, small loader — never import or call the incumbent's ``load_portfolio``.

Real account files (``config/accounts/<account>.yaml``) are gitignored
(Information Disclosure control, T-04-16); ``config/accounts/example.yaml``
is committed with obviously-fake data to document the schema.

Usage::

    from trading_crab_lib.platform.report.holdings import load_account_weights

    account = load_account_weights("glenn_ira")
    # {"weights": {"SPY": 0.4, "TLT": 0.2, ...}, "cash": 0.1}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from trading_crab_lib import CONFIG_DIR

log = logging.getLogger(__name__)

# D-01: weights + cash summing beyond this tolerance of 1.0 logs a WARNING
# but the raw values are still returned — never normalized (T-04-14).
_WEIGHT_SUM_TOLERANCE = 0.02


def load_account_weights(account: str, *, accounts_dir: Path | None = None) -> dict[str, Any]:
    """Load one account's holdings weights + cash from YAML (V5: ``yaml.safe_load`` only).

    Missing file -> neutral dict (``weights={}``, ``cash=1.0``), WARNING,
    never a crash. Weights+cash summing outside ``_WEIGHT_SUM_TOLERANCE`` of
    1.0 -> WARNING, but the RAW (un-normalized) weights/cash are still
    returned — D-01 explicitly forbids silent normalization.

    A non-numeric weight or cash value is coerced via ``try/except
    (TypeError, ValueError)`` (V5 input-validation guard, mirroring the
    incumbent ``load_portfolio()``'s coercion pattern): the bad weight entry
    is skipped with a WARNING; a bad cash value defaults to 0.0 with a
    WARNING. A hostile/malformed file can never execute code or crash the
    report (T-04-15).

    Args:
        account: account name — the file stem under ``accounts_dir``
            (e.g. ``"glenn_ira"`` -> ``glenn_ira.yaml``).
        accounts_dir: overrides ``CONFIG_DIR / "accounts"`` (for tests).

    Returns:
        dict with keys ``"weights"`` (dict[str, float]) and ``"cash"`` (float).
    """
    target_dir = accounts_dir if accounts_dir is not None else (CONFIG_DIR / "accounts")
    path = target_dir / f"{account}.yaml"

    if not path.exists():
        log.warning(
            "Holdings file not found for account %s: %s — treating as empty/neutral", account, path
        )
        return {"weights": {}, "cash": 1.0}

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_weights = data.get("weights", {}) or {}
    weights: dict[str, float] = {}
    for ticker, value in raw_weights.items():
        try:
            weights[ticker] = float(value)
        except (TypeError, ValueError):
            log.warning(
                "Account %s: non-numeric weight for %s (%r) — skipping entry", account, ticker, value
            )

    try:
        cash = float(data.get("cash", 0.0))
    except (TypeError, ValueError):
        log.warning(
            "Account %s: non-numeric cash value (%r) — defaulting to 0.0", account, data.get("cash")
        )
        cash = 0.0

    total = sum(weights.values()) + cash
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        log.warning(
            "Account %s weights+cash sum to %.3f, expected ~1.0 (tolerance %.2f) — "
            "returning RAW values, NOT normalized (D-01)",
            account, total, _WEIGHT_SUM_TOLERANCE,
        )

    return {"weights": weights, "cash": cash}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no real account data (mirrors the
    # platform module __main__ footer convention).
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "demo.yaml").write_text(
            "weights:\n  SPY: 0.5\n  TLT: 0.3\ncash: 0.2\n", encoding="utf-8"
        )
        result = load_account_weights("demo", accounts_dir=tmp_path)
        print(f"self-check: {result}")  # noqa: T201
