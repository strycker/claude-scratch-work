# gsd-salvage — extracted from `gsd-scratch-work` before freezing it

These 5 modules exist **only** in the `gsd-scratch-work` submodule (frozen at
`7b9a21f`, 2026-03-27) and have no equivalent in this repo's `src/trading_crab_lib/`.
Copied verbatim so the ideas survive when the submodule is archived. These are
**reference copies, not wired in** — port the ones marked ✅ when their step is next touched.

| File | Lines | What it does | Port? |
|---|---|---|---|
| `prediction/feature_gating.py` | 64 | Forces `features_supervised.parquet` (causal) for step-5 training; `--allow-noncausal-features` opt-in falls back with a loud warning. | ✅ **Yes** — directly enforces pitfall **P1** (centered features = look-ahead bias). Best value here. |
| `prediction/model_metrics_artifacts.py` | 487 | Multiclass Brier score, calibration bins, tidy confusion tables → JSON/parquet under `outputs/reports/model_metrics/`. Offline audit trail, not training. | ✅ Yes — model-QA gap; complements step-5 monitoring. |
| `prediction/dashboard_model.py` | 55 | Resolves RF vs GB pkl path from `dashboard.regime_model` setting, safe RF fallback. | 🟡 Only if a GB model is ever saved for the dashboard. |
| `ingestion/macro_partial.py` | 107 | Merges only the missing FRED/multpl columns onto a stale `macro_raw` so a partial fetch satisfies step 2 without full `--refresh`. | 🟡 Nice efficiency win; overlaps checkpoint logic. |
| `paths.py` | 136 | `TRADING_CRAB_ROOT` / `_CONFIG` / `_DATA` / `_OUTPUT` env resolution for pip-installed (site-packages) use. | ❌ Superseded — this repo's `__init__.py` already does `TC_*` overrides (ADR D28). Keep only as a more-thorough reference. |

## Freeze status

`gsd-scratch-work` and `trading-crab-lib` submodules are **frozen archives** as of
this commit — everything unique has been extracted here. Do not sync them further;
this repo's `src/trading_crab_lib/` supersedes both. `trading-crab` stays live (the
public/PyPI migration target, per `MIGRATION-PLAN.md`).
