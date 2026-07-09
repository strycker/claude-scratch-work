# PHASE_D_PLAN.md — New Features (Tier 1)

> **⚠️ SUPERSEDED (July 2026).** Historical plan — kept for context, not active work.
> The project's target and execution plan are now `platform_design/platform_design.md`
> (v1.7) and `ROADMAP.md` Tier 0. Do not treat items below as current.

Created: 2026-03-28
Branch: `claude/implement-phase-a2-SmHEA`

---

## Overview

Phase D adds Tier 1 new features that are independent of the packaging restructure.
See `NEXT_STEPS.md` lines 303-316 for the canonical plan.

---

## Tasks

### D1 — Additional FRED Series in Feature Lists

FRED series INDPRO, PAYEMS, DPCERA3Q086SBEA are already defined in
`config/settings.yaml` under `fred.series`. What's missing:

- Add `fred_indpro`, `fred_payems`, `fred_real_pce` to `log_columns`
- Add `log_fred_indpro`, `log_fred_payems`, `log_fred_real_pce` to `initial_features`
- Add derivatives (`_d1`, `_d2`) to `clustering_features`
- These series have deep history (1939+/1939+/1947+), safe for clustering

### D2 — LightGBM Flat-API Integration

**ALREADY DONE.** `train_lightgbm()` exists in `prediction/__init__.py`.
Wired into `run_pipeline.py` step 5. Saves `lightgbm_regime.pkl`.
Added to model comparison. No work needed.

### D3 — Conference Board LEI Proxy

New composite indicator from existing FRED series:
- UNRATE (inverted), T10Y2Y, M2SL, INDPRO, PAYEMS
- Standardize each, equal-weight average → `lei_proxy`
- Add to `initial_features` (not clustering — experimental)
- New module `src/trading_crab_lib/indicators.py`

### D4 — Class-Weight Balancing

`class_weight="balanced"` is already on RF and LGBM. Missing from:
- DecisionTreeClassifier in `train_classifier()` (line 114)
- Forward classifiers in `train_forward_classifiers()` (RF factory)
- Add `prediction.class_balance_method` setting to `config/settings.yaml`
  (values: "balanced", "none"; default "balanced")
- Honor the setting in all classifier factories

---

## Remaining Phases (E–F)

### Phase E — Structural Cleanup
| Task | Description |
|------|-------------|
| E1 | Update MANIFEST.in for both packages |
| E2 | Update CLAUDE.md with 2-package architecture |
| E3 | Update README.md with install instructions |
| E4 | Refactor run_pipeline.py into src/trading_crab/pipeline.py |

### Phase F — Advanced Features (long-term)
| Task | Description |
|------|-------------|
| F1 | Per-asset regime probability models |
| F2 | Walk-forward backtest framework |
| F3 | Interactive Streamlit dashboard |
| F4 | Weekly automated report with AI narrative |
| F5 | Finviz Elite integration |
| F6 | trading-crab-lib config independence |
