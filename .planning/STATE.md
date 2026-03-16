## Project State — Trading-Crab (V1)

## Project Reference

- **Project**: Trading-Crab — Market Regime Analysis & ETF Portfolio Guidance
- **Core value**: Turn macro-driven market regimes and ETF behavior into transparent, regime-aware portfolio recommendations and a weekly advisory-style report.
- **Scope (v1)**: ETF-only universe, weekly/quarterly cadence, recommendation-focused (no auto-trading, no single stocks or direct crypto).

## Current Position

- **Current phase**: Phase 1 — Data & Constraints Foundations
- **Current plan**: Not yet planned (ready for `/gsd:plan-phase 1`)
- **Overall status**: Roadmap created; implementation not started.

### Phase Progress

| Phase | Name                                      | Plans Complete | Status       | Notes                         |
|-------|-------------------------------------------|----------------|--------------|-------------------------------|
| 1     | Data & Constraints Foundations            | 0/0            | Not started  | First planning target         |
| 2     | Regime Clustering & Interpretation        | 0/0            | Not started  | Depends on Phase 1            |
| 3     | Supervised Regime & Behavior Models       | 0/0            | Not started  | Depends on Phases 1–2         |
| 4     | Regime-Conditional ETF & Portfolio Behavior | 0/0          | Not started  | Depends on Phases 1–3         |
| 5     | Recommendations & Machine-Readable Outputs | 0/0          | Not started  | Depends on Phases 1–4         |
| 6     | Weekly Report Pipeline                    | 0/0            | Not started  | Depends on Phases 1–5         |

## Performance & Health

- **Pipeline health**: Unknown (v1 implementation not yet executed end-to-end).
- **Data freshness**: Unknown (to be tracked once ingestion/checkpoints are live).
- **Model performance**: Not yet measured (pending Phases 2–3).

## Accumulated Context

### Key Decisions (from PROJECT.md)

- Focus on ETF-level portfolios only; no single stocks or direct crypto in v1.
- Allow bitcoin exposure only via ETF wrappers.
- Weekly report cadence; regime focus is quarterly.

### Open Questions / Risks

- How stable are regime labels over time as new data arrives?
- How robust are supervised models and portfolio templates to regime changes outside historical experience?

### Working Notes

- Use `ROADMAP.md` as the source of truth for phase goals, dependencies, and success criteria.
- Use `REQUIREMENTS.md` to keep requirement IDs and traceability aligned as implementation proceeds.

