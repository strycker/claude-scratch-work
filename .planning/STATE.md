## Project State — Trading-Crab (V1)

## Project Reference

- **Project**: Trading-Crab — Market Regime Analysis & ETF Portfolio Guidance
- **Core value**: Turn macro-driven market regimes and ETF behavior into transparent, regime-aware portfolio recommendations and a weekly advisory-style report.
- **Scope (v1)**: ETF-only universe, weekly/quarterly cadence, recommendation-focused (no auto-trading, no single stocks or direct crypto).

## Current Position

- **Current phase**: Phase 3 — Supervised Regime & Behavior Models
- **Current plan**: 03-01 (awaiting execution — no SUMMARY exists)
- **Overall status**: Phases 1-2 complete; Phase 3 in progress (2/3 plans done).

### Phase Progress

| Phase | Name                                      | Plans Complete | Status       | Notes                         |
|-------|-------------------------------------------|----------------|--------------|-------------------------------|
| 1     | Data & Constraints Foundations            | 3/3            | Complete     | All 3 plans have SUMMARYs     |
| 2     | Regime Clustering & Interpretation        | 2/2            | Complete     | Marked complete 2026-03-16    |
| 3     | Supervised Regime & Behavior Models       | 2/3            | In Progress  | Plans 02+03 done; 01 pending  |
| 4     | Regime-Conditional ETF & Portfolio Behavior | 0/0          | Not started  | Depends on Phases 1–3         |
| 5     | Recommendations & Machine-Readable Outputs | 0/0          | Not started  | Depends on Phases 1–4         |
| 6     | Weekly Report Pipeline                    | 0/0            | Not started  | Depends on Phases 1–5         |

## Performance & Health

- **Pipeline health**: All 7 steps run end-to-end. 238 tests collected.
- **Data freshness**: Checkpoints in `data/checkpoints/`; re-scrape with `--refresh`.
- **Model performance**: RF + DT trained with TSCV; forward classifiers operational.

## Accumulated Context

### Key Decisions (from PROJECT.md)

- Focus on ETF-level portfolios only; no single stocks or direct crypto in v1.
- Allow bitcoin exposure only via ETF wrappers.
- Weekly report cadence; regime focus is quarterly.
- `prediction/__init__.py` flat API is the pipeline interface; `prediction/classifier.py` bundle API is test-only (ADR #12).
- Balanced clustering (`balanced_cluster`) is the default for downstream regime labeling.

### Open Questions / Risks

- How stable are regime labels over time as new data arrives?
- How robust are supervised models and portfolio templates to regime changes outside historical experience?

### Working Notes

- Use `ROADMAP.md` as the source of truth for phase goals, dependencies, and success criteria.
- Use `REQUIREMENTS.md` to keep requirement IDs and traceability aligned as implementation proceeds.
- Phase 3 plan 01 is the next plan to execute (no SUMMARY exists for it yet).

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260317-3ng | refresh and expand ROADMAP.md, README.md, STATE.md, and agent docs to capture current structure, models, and next steps | 2026-03-17 | c3886e3 | [260317-3ng-refresh-and-expand-roadmap-md-readme-md-](./quick/260317-3ng-refresh-and-expand-roadmap-md-readme-md-/) |

**Last activity:** 2026-03-17 - Completed quick task 260317-3ng: refresh and expand ROADMAP.md, README.md, STATE.md, and agent docs
