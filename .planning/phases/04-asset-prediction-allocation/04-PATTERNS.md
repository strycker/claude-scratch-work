# Phase 4: Asset Prediction & Allocation - Pattern Map

**Mapped:** 2026-07-22
**Files analyzed:** 14 (10 new modules + 2 config edits + 1 config file + 1 test dir)
**Analogs found:** 13 / 14

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `platform/assets/returns.py` | service (analytics) | batch/transform | `platform/prediction/transition_matrix.py` (pure compute + `__main__`) + incumbent `asset_returns.py` (inspiration only) | role-match |
| `platform/assets/vol.py` | service (analytics) | transform | `platform/honesty/gap_lag.py` (compute_* + report_* + `__main__`) | exact (shape) |
| `platform/allocation/hysteresis.py` | service (state machine) | event-driven | `platform/labeling/diagnostics.py` (`_churn_against_previous` load-before-save pattern) | exact |
| `platform/allocation/tilt.py` | service (compute) | transform | `platform/honesty/gap_lag.py` | exact (shape) |
| `platform/tripwire/monitor.py` | CLI/service | event-driven (OR-logic) | `platform/honesty/gap_lag.py` (`__main__` CLI + report_*) | role-match |
| `platform/report/holdings.py` | config loader / validator | request-response | `platform/config.py` (`load_platform_config`, defensive `.get()`, collect-all-errors) | exact |
| `platform/report/weekly.py` | report/delivery | batch + request-response | `platform/labeling/diagnostics.py` (orchestration wiring) + incumbent `email.py` (reused, read-only) | role-match |
| `platform/ingestion/prices_daily.py` (MODIFY: add SPY) | ingestion | batch fetch | itself — no analog needed, one-line config addition | n/a (existing file, config-only change) |
| `config/platform_settings.yaml` (MODIFY: `universe.satellites` +SPY, new `fred_daily` DAAA/DBAA, `allocation:`/`tripwire:`/`report:` sections) | config | n/a | itself — existing `fred_monthly`/`universe` blocks are the template | exact |
| `platform/ingestion/macro_daily.py` (NEW — DAAA/DBAA fetch) | ingestion | batch fetch | `platform/ingestion/macro_monthly.py` (`_fetch_fred_monthly` skeleton, minus the resample) | exact (shape, no-resample analog) |
| `tests/unit/test_platform_returns.py` | test | — | `tests/unit/test_platform_transition_matrix.py`, `test_platform_gap_lag.py` | exact |
| `tests/unit/test_platform_vol.py` | test | — | `tests/unit/test_platform_gap_lag.py` | exact |
| `tests/unit/test_platform_hysteresis.py` | test | — | `tests/unit/test_platform_labeling.py` (churn/load-before-save tests) | exact |
| `tests/unit/test_platform_tripwire.py` | test | — | `tests/unit/test_platform_gap_lag.py` | exact |

No analog exists for `report/weekly.py`'s exact "assemble markdown + call incumbent email" shape inside `platform/` (nothing in `platform/` sends email yet) — see "No Analog Found" below.

## Pattern Assignments

### `platform/assets/returns.py` (service, batch/transform)

**Analog:** `platform/prediction/transition_matrix.py` (module shape) + incumbent `src/trading_crab_lib/asset_returns.py` (inspiration only per D-01 — do not import)

**Imports pattern** (from `transition_matrix.py` lines 21-23):
```python
from __future__ import annotations

import pandas as pd
```

**Core pattern — pure compute function + `__main__` self-check** (`transition_matrix.py` lines 26-48):
```python
def empirical_transition_matrix(states: pd.Series) -> pd.DataFrame:
    """Row-normalized K x K count table..."""
    pairs = pd.DataFrame({"from": states.iloc[:-1].values, "to": states.iloc[1:].values})
    counts = pd.crosstab(pairs["from"], pairs["to"])
    return counts.div(counts.sum(axis=1), axis=0)

if __name__ == "__main__":
    demo_states = pd.Series([0, 1, 0, 1, 1, 0, 2, 2, 1])
    demo_matrix = empirical_transition_matrix(demo_states)
    print(demo_matrix)  # noqa: T201
```

**NULL-tolerant per-(regime, asset) stats pattern** (RESEARCH.md's own worked example, adapted from incumbent `asset_returns.py::returns_full_stats` — do not import, D-01):
```python
def returns_by_regime_stats(returns: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    common = returns.index.intersection(states.index)
    joined = returns.loc[common].copy()
    joined["regime"] = states.loc[common]
    rows = []
    for regime, group in joined.groupby("regime"):
        for ticker in group.drop(columns=["regime"]).columns:
            col = group[ticker].dropna()
            if col.empty:
                continue
            cum = (1 + col).cumprod()
            rows.append({
                "regime": regime, "asset": ticker,
                "mean_monthly_return": col.mean(), "std_monthly_return": col.std(),
                "sharpe_annualized": (col.mean() / col.std()) * (12 ** 0.5) if col.std() > 0 else float("nan"),
                "hit_rate": (col > 0).mean(),
                "max_drawdown": (cum / cum.cummax() - 1).min(),
                "n_obs": len(col),
            })
    return pd.DataFrame(rows)
```

**Error handling:** none needed beyond `.dropna()`/`empty` guards — no I/O in the compute function (I/O lives only in `report_*`/loader code per Pattern 1).

---

### `platform/assets/vol.py` (service, transform)

**Analog:** `platform/honesty/gap_lag.py` (full module shape)

**Imports pattern** (`gap_lag.py` lines 21-31):
```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from trading_crab_lib import OUTPUT_DIR

log = logging.getLogger(__name__)
```

**Core EWMA pattern** (RESEARCH.md Pattern 1, matches `gap_lag.py`'s pure-function style):
```python
def ewma_vol(returns: pd.Series, *, halflife: float, annualization_factor: float) -> pd.Series:
    """Annualized EWMA volatility. halflife in the SAME units as returns' index."""
    return returns.ewm(halflife=halflife, min_periods=2).std() * np.sqrt(annualization_factor)
```

**Artifact persistence pattern — schema-stable columns, empty-safe** (`gap_lag.py` lines 33-36, 109-146):
```python
_ARTIFACT_COLUMNS = ["asset", "ewma_vol_annualized", "halflife", "as_of_date"]

def report_vol(metrics: dict, *, output_dir: Path | None = None) -> Path:
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "platform"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "ewma_vol.parquet"
    rows = [...] if metrics else []
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_ARTIFACT_COLUMNS)
    df.to_parquet(artifact_path, index=False)
    log.info(...)
    print(...)  # noqa: T201 — first-class CLI output, not debug noise
    return artifact_path
```

**`__main__` self-check pattern** (`gap_lag.py` lines 149-173) — synthetic series, no network, no checkpoint dependency.

---

### `platform/allocation/hysteresis.py` (service, state machine)

**Analog:** `platform/labeling/diagnostics.py` (`_get_checkpoint_manager` + `_churn_against_previous` load-before-save pattern)

**Imports pattern** (`diagnostics.py` lines 44-56, trimmed to what hysteresis needs):
```python
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

log = logging.getLogger(__name__)
```

**Load-before-save state pattern** (`diagnostics.py` lines 159-177 — critical ordering, Pitfall 3 there applies identically here):
```python
def _get_checkpoint_manager(checkpoint_dir: Path | None) -> CheckpointManager:
    if checkpoint_dir is None:
        return get_platform_checkpoint_manager()
    return CheckpointManager(checkpoint_dir=checkpoint_dir)

def load_active_regime(cm=None) -> int | None:
    cm = cm or get_platform_checkpoint_manager()
    try:
        state = cm.load("hysteresis_state")
    except FileNotFoundError:
        return None  # cold start
    return int(state["active_regime"].iloc[0])

def save_active_regime(active_regime: int | None, cm=None) -> None:
    cm = cm or get_platform_checkpoint_manager()
    cm.save(pd.DataFrame([{"active_regime": active_regime}]), "hysteresis_state")
```

**State-machine logic** (RESEARCH.md Pitfall 3 — track P(active regime), never argmax; exact 3-branch semantics quoted there; also see the D-04-style config-driven thresholds pattern in `config.py`'s `.get()` reads).

**Error handling:** `FileNotFoundError` on cold start is the ONLY expected exception (matches `diagnostics.py`'s `except FileNotFoundError: return float("nan")` idiom) — no bare except.

**Test pattern to mirror:** `tests/unit/test_platform_labeling.py` — look for its churn / load-before-save test cases as the template for the no-flip invariant test.

---

### `platform/allocation/tilt.py` (service, transform)

**Analog:** `platform/honesty/gap_lag.py` (compute-function shape) + RESEARCH.md's own worked `vol_target_scale()`/`portfolio_vol()` examples (already codebase-grounded, use verbatim):

```python
def vol_target_scale(target_vol_annual: float, portfolio_vol_annual: float) -> float:
    """Long-only, no-leverage scale factor. Cash absorbs 1 - scale."""
    if portfolio_vol_annual <= 0:
        return 0.0
    return min(1.0, target_vol_annual / portfolio_vol_annual)

# ponytail: linear-sum-of-vols fallback deliberately ignores diversification —
# it can only make the tilt MORE conservative than the true portfolio vol, never
# less. Upgrade path: regime-conditional Ledoit-Wolf covariance (L3-V2-01, v2).
def portfolio_vol(weights: pd.Series, asset_returns: pd.DataFrame, *, halflife: float,
                   min_obs: int = 12) -> float:
    common_assets = weights.index.intersection(asset_returns.columns)
    aligned = asset_returns[common_assets].dropna()
    if len(aligned) >= min_obs:
        port_returns = (aligned * weights[common_assets]).sum(axis=1)
        return float(ewma_vol(port_returns, halflife=halflife, annualization_factor=12).iloc[-1])
    per_asset_vol = asset_returns[common_assets].apply(
        lambda s: ewma_vol(s.dropna(), halflife=halflife, annualization_factor=12).iloc[-1]
    )
    return float((weights[common_assets].abs() * per_asset_vol).sum())
```

Import `ewma_vol` from `platform/assets/vol.py` (do not duplicate the EWMA function).

---

### `platform/tripwire/monitor.py` (CLI/service, event-driven OR-logic)

**Analog:** `platform/honesty/gap_lag.py` (`__main__` CLI + `report_*` shape)

**Escalation enum + OR-logic pattern** (RESEARCH.md worked example, codebase-consistent style — `str, Enum` matches project convention of typed constants):
```python
from enum import Enum

class TripwireEscalation(str, Enum):
    NONE = "none"
    RUN_WEEKLY_SCORING_EARLY = "run weekly scoring early"
    TIER1_DERISK_REVIEW = "Tier-1 de-risk review"

def escalate(vol_spike: bool, credit_velocity: bool, spy_drawdown: bool) -> TripwireEscalation:
    n_triggered = sum([vol_spike, credit_velocity, spy_drawdown])
    if n_triggered >= 2:
        return TripwireEscalation.TIER1_DERISK_REVIEW
    if n_triggered == 1:
        return TripwireEscalation.RUN_WEEKLY_SCORING_EARLY
    return TripwireEscalation.NONE
```

**`__main__` CLI entry point pattern** (`gap_lag.py` lines 149-173, adapted for `python3 -m trading_crab_lib.platform.tripwire.monitor` per D-04): synthetic self-check block plus `logging.basicConfig(level=logging.INFO)`; print the escalation enum's value as the last line of CLI output (mirrors `gap_lag.py`'s `print(summary)  # noqa: T201`).

**Data dependency:** needs SPY daily closes (`daily_raw["SPY"]`) and DAAA/DBAA daily spread — see Wave 0 ingestion changes below.

---

### `platform/report/holdings.py` (config loader/validator)

**Analog:** `platform/config.py` (`load_platform_config` / `validate_platform_config` — collect-all-errors, defensive `.get()`, tri-mode input)

**Imports pattern** (`config.py` lines 12-24):
```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from trading_crab_lib import CONFIG_DIR

log = logging.getLogger(__name__)
```

**Warn-don't-fail validation pattern** (adapted from `config.py`'s collect-all-errors style but WARNING not raise, per D-01 explicitly rejecting incumbent `load_portfolio()`'s silent-normalize):
```python
_WEIGHT_SUM_TOLERANCE = 0.02  # D-01: validate with tolerance, warn don't fail

def load_account_weights(account: str, *, accounts_dir: Path | None = None) -> dict[str, Any]:
    """yaml.safe_load() only (never yaml.load — V5 security convention).
    Returns {"weights": {ticker: frac}, "cash": frac}. Missing file -> empty/
    neutral dict, not a crash (D-01)."""
    target_dir = accounts_dir or (CONFIG_DIR / "accounts")
    path = target_dir / f"{account}.yaml"
    if not path.exists():
        log.warning("Holdings file not found for account %s: %s — treating as empty/neutral", account, path)
        return {"weights": {}, "cash": 1.0}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    weights = data.get("weights", {})
    cash = float(data.get("cash", 0.0))
    total = sum(weights.values()) + cash
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        log.warning("Account %s weights+cash sum to %.3f, expected ~1.0 (tolerance %.2f)",
                    account, total, _WEIGHT_SUM_TOLERANCE)
    return {"weights": weights, "cash": cash}
```

**Explicit anti-pattern (from RESEARCH.md "Anti-Patterns to Avoid" + "Don't Hand-Roll"):** do NOT import or call `trading_crab_lib.config.load_portfolio()` — it silently normalizes to sum=1 and has no cash concept, conflicting with D-01.

**Security note (V5 Input Validation):** `yaml.safe_load()` only; numeric coercion wrapped in `try/except (TypeError, ValueError)`, mirroring incumbent `load_portfolio()`'s existing pattern per RESEARCH.md's Security Domain section.

---

### `platform/report/weekly.py` (report assembly + delivery)

**Analog:** `platform/labeling/diagnostics.py`'s `label_regimes()` orchestration shape (pull -> compute -> persist -> report) for the assembly half; incumbent `src/trading_crab_lib/email.py` for delivery (READ-ONLY reuse, verified signatures).

**Delivery pattern — verified exact signatures** (`email.py` lines 135-171, 288-330):
```python
from pathlib import Path
from trading_crab_lib.email import build_weekly_email_body, load_email_config, send_weekly_email

platform_report_dir = Path("outputs/reports/platform")
# weekly.py has already written platform_report_dir / "weekly_report.md"

subject, body = build_weekly_email_body(
    platform_report_dir, subject_prefix="Trading-Crab Platform Weekly Report"
)
if args.send_email:
    cfg = load_email_config()
    if cfg:
        send_weekly_email(cfg, subject, body)  # plot_paths=None in v1
```

`build_weekly_email_body(report_dir, subject_prefix)` checks `email_body.txt` -> `weekly_report.md` -> `dashboard.csv` in that order and reads the markdown file **verbatim** — no markdown parsing needed on the platform side. `send_weekly_email(config, subject, body, plot_paths=None)` returns `bool`; validates `REQUIRED_KEYS` against `config` and logs `ERROR` (not raise) on missing keys — matches this phase's "email is opt-in, degrade gracefully" framing (D-02).

**Artifact path convention:** `outputs/reports/platform/weekly_report.md` — distinct from incumbent's `outputs/reports/weekly_report.md` (D-02, avoids collision). Reuse `OUTPUT_DIR` from `trading_crab_lib` (see `gap_lag.py` line 29) plus `/ "reports" / "platform"` (matches `report_labeling_diagnostics`'s and `report_gap_lag`'s `target_dir` pattern, lines 217/126).

**Trades-implied table** (RESEARCH.md's own worked example, inspiration from incumbent `reporting.py::generate_recommendation` — do NOT import into `platform/`, D-01):
```python
def trades_implied(target_weights: pd.Series, current_weights: pd.Series,
                    threshold: float = 0.03) -> pd.DataFrame:
    """One row per asset: current_pct, target_pct, delta_pct, signal (BUY/SELL/HOLD).
    threshold is a flat no-trade band (design §21 full band-width math is L4-V2-01)."""
    ...
```

---

### `platform/ingestion/macro_daily.py` (NEW — Wave 0 gap-fill for DAAA/DBAA)

**Analog:** `platform/ingestion/macro_monthly.py`'s `_fetch_fred_monthly` (lines 75-93) — same client construction, minus the resample.

**Core pattern (no resample — daily preserved, mirrors `prices_daily.py`'s "no internal resample" doctrine)**:
```python
from fredapi import Fred

def _fetch_fred_daily(fred: Fred, series_id: str, start: str, end: str) -> pd.Series:
    """Pull one FRED series at native (daily) frequency — no resample, unlike
    _fetch_fred_monthly (macro_monthly.py:75-93). DAAA/DBAA are already daily."""
    return fred.get_series(series_id, observation_start=start, observation_end=end)
```

Reuse the same `ThreadPoolExecutor` + try/except-WARNING skeleton from `fetch_fred_monthly` (`macro_monthly.py` lines 96-158) for the orchestrator (`fetch_fred_daily(cfg)`), reading a new `cfg["fred_daily"]["series"]` config block (mirrors `fred_monthly.series` shape). Persist under the platform checkpoint namespace (e.g. merged into or alongside `daily_raw`).

**Config addition** (`config/platform_settings.yaml`, template = existing `fred_monthly.series` block, lines 20-48):
```yaml
fred_daily:
  series:
    DAAA:
      name: "fred_daa_daily"
      shift: false
    DBAA:
      name: "fred_dbaa_daily"
      shift: false
```

**Config addition — SPY** (`config/platform_settings.yaml`, `universe.satellites`, line 137 — one-line addition per Pitfall 1):
```yaml
  satellites:
    - SPY   # dual-purpose: tradable proxy (already documented in splice.equities.tradable)
            # AND tripwire drawdown-from-peak input (Phase 4, Pitfall 1)
    - QQQ
    ...
```

## Shared Patterns

### Config sections — defensive `.get()`, never added to required list
**Source:** `platform/config.py` lines 26-33 (`_REQUIRED_PLATFORM_SECTIONS`) + module docstring
**Apply to:** every new `allocation:`/`tripwire:`/`report:` config read across all Phase 4 modules
```python
allocation_cfg = cfg.get("allocation", {})
target_vol = allocation_cfg.get("target_vol_annual", 0.10)
```
Do NOT add `allocation`/`tripwire`/`report` to `_REQUIRED_PLATFORM_SECTIONS` in `config.py` (RESEARCH.md explicit constraint) — these are optional, defaulted sections.

### Module shape — compute + report + `__main__` self-check
**Source:** `platform/honesty/gap_lag.py` (canonical), also `platform/prediction/transition_matrix.py`, `platform/labeling/diagnostics.py`
**Apply to:** every new file in `platform/assets/`, `platform/allocation/`, `platform/tripwire/`
Pure `compute_*()` functions (no I/O) → one `report_*()`/persist function (schema-stable parquet + human-readable `print`/`log.info`) → synthetic no-network `__main__` self-check block.

### Checkpoint persistence — never hand-roll save/load
**Source:** `platform/checkpoints.py` (`get_platform_checkpoint_manager()`) — thin wrapper over incumbent `CheckpointManager`
**Apply to:** hysteresis state, any new persisted artifact (returns-by-regime table, vol series)
```python
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
cm = get_platform_checkpoint_manager()
cm.save(df, "returns_by_regime")
df = cm.load("returns_by_regime")
```

### Load-before-save ordering for any persisted state
**Source:** `platform/labeling/diagnostics.py` lines 166-177 (`_churn_against_previous`, "Pitfall 3")
**Apply to:** `platform/allocation/hysteresis.py` exclusively — MUST load the previous `hysteresis_state` checkpoint before writing the new one, or the state machine silently reads its own just-written value.

### Email delivery — reuse incumbent, never fork
**Source:** `src/trading_crab_lib/email.py` (read-only)
**Apply to:** `platform/report/weekly.py` only
`build_weekly_email_body()` / `load_email_config()` / `send_weekly_email()` — exact signatures verified above under Pattern for `weekly.py`.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `platform/report/weekly.py`'s full "assemble markdown across multiple upstream sources (regime dist + trajectory + per-asset signals + trades-implied)" orchestration | report/delivery | batch | No existing `platform/` module assembles a multi-section markdown report; nearest is incumbent `reporting.py::print_dashboard`/`generate_recommendation`, inspiration-only (not imported, D-01). Use design §7's output list + `label_regimes()`'s pull→compute→persist→report shape as the structural template instead. |

## Metadata

**Analog search scope:** `src/trading_crab_lib/platform/` (all 24 .py files), `src/trading_crab_lib/email.py`, `src/trading_crab_lib/asset_returns.py`/`reporting.py` (inspiration-only per D-01), `config/platform_settings.yaml`, `tests/unit/test_platform_*.py` (16 files)
**Files scanned:** ~30
**Pattern extraction date:** 2026-07-22
