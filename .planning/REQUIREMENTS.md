## Trading-Crab — V1 Requirements

### Overview

This document captures the v1 functional and non-functional requirements for Trading-Crab, a market-regime analysis and ETF portfolio guidance system.

- Scope: ETF-only portfolios (broad indices, sectors, bonds, precious metals, bitcoin ETF).
- Cadence: Weekly / quarterly, not intraday.
- Outputs: Research views + concrete buy/hold/sell recommendations and a weekly report.

Each requirement has a stable ID for traceability from roadmap → plans → tests.

---

## v1 Requirements

### 1. Data Ingestion & Feature Engineering

- **DATA-01 — Macro & ETF ingestion**
  - **Description**: Ingest macroeconomic time series (multpl + FRED) and ETF prices for the chosen ETF universe.
  - **Details**:
    - Macro data: quarterly resolution, ~1950–present where available.
    - ETFs: as far back as each ETF’s history allows.
  - **Rationale**: Regime discovery and ETF behavior analysis require long history.

- **DATA-02 — Checkpointed pipeline**
  - **Description**: Maintain a checkpointed data pipeline so full re-scrapes are optional.
  - **Details**:
    - Use parquet checkpoints + manifest to avoid re-scraping on every run.
    - Typical workflows should re-use cached data/checkpoints.
  - **Rationale**: Scraping is slow; iteration must be fast.

- **DATA-03 — Feature set & causal variants**
  - **Description**: Compute a stable, documented feature set with both non-causal and causal variants.
  - **Details**:
    - Include cross-asset ratios, log transforms, Bernstein gap fill, smoothed derivatives, and any additional yield-curve/feature work defined in `settings.yaml`.
    - Ensure any supervised training that mimics “what was knowable then” uses causal features with no look-ahead bias.
  - **Rationale**: Prevent leakage while preserving analytical richness.

### 2. Regime Clustering & Interpretation

- **REGIME-01 — Regime clustering**
  - **Description**: Cluster quarters into a small, interpretable set of regimes.
  - **Details**:
    - Use PCA + KMeans (and related tools already in the codebase).
    - Target ~4–7 regimes with clear differences in macro conditions and asset returns.
  - **Rationale**: Regimes are the core abstraction for all downstream logic.

- **REGIME-02 — Regime profiling**
  - **Description**: For each regime, compute descriptive statistics over key macro features and ETF returns.
  - **Details**:
    - Macro profiles are produced via `regime.build_profiles()` and saved to `data/regimes/profiles.parquet`.
    - ETF return behavior by regime is captured in `data/regimes/asset_return_profile.parquet` (written by step 6 `06_asset_returns.py` via `returns_by_regime()`).
    - Summaries should enable human-readable descriptions (e.g. “high inflation, low growth, tight credit”) and be reproducible from code (not manual notebooks only).
  - **Rationale**: Users need to understand what each regime actually means, both in macro terms and in terms of ETF performance.

- **REGIME-03 — Stable regime naming**
  - **Description**: Provide a stable mapping from cluster IDs to human-readable regime names.
  - **Details**:
    - Keep the mapping in version-controlled config (e.g. `regime_labels.yaml`).
    - Document any renames / re-labeling decisions.
  - **Rationale**: Names should not silently shift across runs; critical for longitudinal analysis.

### 3. Supervised Models (Current & Forward)

- **MODEL-01 — Current-regime classifier**
  - **Description**: Train models that predict the current quarter’s regime from currently-available features.
  - **Details**:
    - No forward-looking features; respect publication lags and causal variants.
    - Use time-series appropriate validation (e.g. `TimeSeriesSplit` walk-forward).
  - **Rationale**: Needed to turn unsupervised regimes into a real-time signal.

- **MODEL-02 — Forward regime transitions**
  - **Description**: Train forward-horizon models that estimate regime transition probabilities.
  - **Details**:
    - At minimum, support 1-quarter-ahead transitions; ideally extend to 2–8 quarters.
    - Outputs should be interpretable probabilities per future regime (given current regime and features).
  - **Rationale**: Portfolio guidance should account for likely near-term regime shifts.

- **MODEL-03 — Forward ETF/portfolio behavior**
  - **Description**: Provide simple forward-looking behavior estimates for individual ETFs and candidate portfolios.
  - **Details**:
    - At minimum, a directional view (up/flat/down or similar) for the next quarter.
    - Prefer models that can be summarized in a small set of metrics (probability of “good” vs “bad” outcome).
  - **Rationale**: Helps connect regime predictions to asset-level expectations.

- **MODEL-04 — Evaluation & reporting**
  - **Description**: Evaluate supervised models with transparent metrics and confusion-style summaries.
  - **Details**:
    - Track performance across train/test splits with time-series aware CV.
    - Produce human-inspectable metrics (accuracy, F1, etc.) and per-class breakdowns where meaningful.
  - **Rationale**: User needs to see whether models are actually working before trusting recommendations.

### 4. ETF & Portfolio Behavior by Regime

- **PORT-01 — Regime-conditional ETF returns**
  - **Description**: For each regime, compute historical return distributions for every ETF in scope.
  - **Details**:
    - Include median, quantiles, and basic risk metrics (e.g. drawdowns) per regime.
    - Surface which ETFs are generally “green” or “red” in each environment.
  - **Rationale**: Forms the empirical basis for regime-aware allocations.

- **PORT-02 — Portfolio templates**
  - **Description**: Define and evaluate a small library of simple portfolio templates.
  - **Details**:
    - Example templates: 60/40, risk-off, risk-on, barbell, inflation-hedged.
    - Compute regime-conditional performance for each template.
  - **Rationale**: Provides understandable building blocks for recommendations.

- **PORT-03 — Current-portfolio expectation**
  - **Description**: Given a specific ETF portfolio (weights), estimate expected return and risk under:
    - The inferred current regime.
    - The most likely near-term regimes (from transition models).
  - **Rationale**: Bridges the gap between high-level regimes and the user’s actual holdings.

### 5. Recommendations & UX

- **UX-01 — Incremental ETF recommendations**
  - **Description**: Generate concrete per-ETF buy/hold/sell recommendations for a given current portfolio.
  - **Details**:
    - Recommendations should move weights incrementally toward a regime-aware target mix.
    - Respect ETF-only constraints (no single stocks, no direct crypto).
    - Avoid hyperactive turnover; changes should be plausible to implement weekly.
  - **Rationale**: Turn analysis into actionable guidance without over-trading.

- **UX-02 — Explanation & transparency**
  - **Description**: For each recommendation, provide a brief explanation grounded in regime and ETF behavior.
  - **Details**:
    - E.g. “Reduce `TLT` because current regime and likely transitions historically penalize long-duration bonds.”
  - **Rationale**: User must understand the “why” behind suggestions to trust them.

- **UX-03 — Machine-readable outputs**
  - **Description**: Emit machine-readable artifacts (CSV/JSON) capturing:
    - Current regime and probabilities.
    - Key transition probabilities.
    - Current portfolio metrics.
    - Recommended trades / target weights.
  - **Rationale**: Enables downstream automation and experimentation (dashboards, emails, scripts).

### 6. Weekly Report / Automation

- **REPORT-01 — Weekly report pipeline**
  - **Description**: Provide a reproducible way (CLI or small script) to generate a full weekly report.
  - **Details**:
    - Refresh or reuse data as configured.
    - Infer current regime and key transitions.
    - Compute current-portfolio expectations.
    - Generate recommendations and a human-readable summary.
  - **Rationale**: Encodes the “one-button” workflow for ongoing use.

- **REPORT-02 — Email-ready summary**
  - **Description**: Produce a compact text summary suitable to send as a weekly email.
  - **Details**:
    - Include: current regime (with confidence), notable transition risks, and ETF-level buy/hold/sell suggestions.
    - Actual email sending can remain out-of-scope; focus on the content.
  - **Rationale**: Aligns with the end vision of an automated weekly advisory email.

### 7. Constraints & Non-Goals

- **CONSTR-01 — ETF-only universe**
  - **Description**: The system must operate on ETFs only for v1.
  - **Details**:
    - Broad indices, sectors, bonds, precious metals, and a bitcoin ETF are in scope.
    - No individual stocks or direct cryptocurrency holdings.
  - **Rationale**: Keeps the problem tractable and aligned with current data/architecture.

- **CONSTR-02 — No intraday / auto-trading**
  - **Description**: The system must not attempt intraday or fully automated trade execution.
  - **Details**:
    - Cadence: weekly/quarterly; no reliance on sub-daily data.
    - Outputs are recommendations and reports, not broker orders.
  - **Rationale**: Focus v1 on robust research and guidance, not execution infrastructure.

---

## Traceability Notes

- `PROJECT.md` defines the vision, scope boundaries, and high-level outcomes.
- This file (`REQUIREMENTS.md`) refines that vision into concrete, testable v1 requirements.
- `ROADMAP.md` will map these requirements to phases; each phase will reference specific IDs (e.g. DATA-01, REGIME-01).
- Phase plans and tests should always reference requirement IDs to keep alignment explicit.

## Traceability

| Requirement | Phase  | Status  |
|------------|--------|---------|
| DATA-01    | Phase 1 | Complete |
| DATA-02    | Phase 1 | Complete |
| DATA-03    | Phase 1 | Complete |
| REGIME-01  | Phase 2 | Complete |
| REGIME-02  | Phase 2 | Complete |
| REGIME-03  | Phase 2 | Complete |
| MODEL-01   | Phase 3 | Pending |
| MODEL-02   | Phase 3 | Pending |
| MODEL-03   | Phase 3 | Pending |
| MODEL-04   | Phase 3 | Pending |
| PORT-01    | Phase 4 | Pending |
| PORT-02    | Phase 4 | Pending |
| PORT-03    | Phase 4 | Pending |
| UX-01      | Phase 5 | Pending |
| UX-02      | Phase 5 | Pending |
| UX-03      | Phase 5 | Pending |
| REPORT-01  | Phase 6 | Pending |
| REPORT-02  | Phase 6 | Pending |
| CONSTR-01  | Phase 1 | Complete |
| CONSTR-02  | Phase 1 | Complete |

