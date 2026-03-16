## Project: Trading-Crab — Market Regime Analysis & ETF Portfolio Guidance

### Purpose

Trading-Crab is a research and decision-support system that analyzes macroeconomic time series and ETF performance to:

- Classify historical and current quarters into market regimes (e.g. stagflation, growth boom).
- Understand how broad asset-class and sector ETFs behave in each regime.
- Predict the current quarter’s regime and expected portfolio return.
- Provide forward-looking guidance and weekly buy/hold/sell recommendations for a real ETF portfolio.

This repository is also a scratch/workbench for experimenting with Claude Code–assisted development; Trading-Crab is the primary, production-quality project inside it.

### Target Users

- A single primary user (the repository owner) acting as:
  - Quantitatively minded investor.
  - Developer comfortable running Python pipelines and notebooks.

The system may later be exposed to additional users (e.g. via a dashboard), but v1 is optimized for solo use.

### V1 Outcome (Definition of Success)

For v1, success means:

1. **Historical & current regime analysis**
   - The system can ingest ~1950–present macroeconomic data at quarterly resolution.
   - It can cluster quarters into a manageable number of interpretable regimes and label them with human-readable names.
   - It can analyze past and current financial quarters to characterize “what kind of environment we’re in” (growth, inflation, stress, etc.).

2. **Regime-aware ETF performance understanding**
   - For each regime, the system can report which ETFs and simple portfolios historically performed well/poorly.
   - It can rank ETFs and a small set of portfolio templates by regime-conditional returns and risk characteristics.

3. **Unsupervised + supervised learning pipeline**
   - Unsupervised models (e.g. PCA + clustering) establish regime labels without look-ahead bias.
   - Supervised models can:
     - Predict the current quarter’s regime from currently-available features.
     - Estimate expected returns for the current ETF portfolio under the inferred current regime.

4. **Forward-looking asset and portfolio predictions**
   - Models can estimate, for at least the next quarter:
     - Likely regime transitions (transition probabilities from current regime to others).
     - Directional behavior (and ideally rough distribution) of key ETFs and candidate portfolios.

5. **Actionable portfolio recommendations**
   - Given the current ETF portfolio and the current/forecast regimes, the system can:
     - Propose incremental buy/sell/hold changes that move the portfolio toward an “ideal” regime-aware target mix.
     - Explain recommendations at the ETF/asset-class level (not black-box).

6. **Weekly email/report**
   - Once per week, the system can generate a concise report that includes:
     - The inferred current regime (with confidence).
     - Notable regime-transition risks for the next few quarters.
     - A simple, actionable buy/hold/sell summary for the current ETF portfolio.
   - The report should be suitable for sending as an email (text first; future automation is a stretch goal).

If all of the above are working end-to-end on real data with reasonable performance and tests, v1 is considered successful.

### In-Scope Assets (V1)

**V1 restricts itself strictly to ETF-level exposure**:

- Broad equity indices (e.g. S&P 500 via `SPY` or similar).
- Sector and style ETFs (e.g. tech vs consumer discretionary).
- Precious metals and related miners (e.g. `IAU`, `SLV`, `GDX`).
- Bonds and duration exposure (e.g. `TLT`, other Treasury or aggregate bond ETFs).
- A **bitcoin ETF** for crypto exposure (spot or futures-based), treated like any other ETF.

All models, dashboards, and recommendations should treat these as the core investable universe for v1.

### Explicitly Out of Scope (V1)

The following are **deliberately excluded** from v1:

- Individual stocks (single-name equity selection).
- Direct cryptocurrency holdings (e.g. on-chain BTC/ETH); only ETF wrappers are allowed.
- Options, futures beyond what’s embedded in ETFs, leverage, and short-selling strategies.
- Intraday or high-frequency trading; the cadence is weekly / quarterly, not minute-by-minute.
- Fully automated trade execution; v1 focuses on recommendations, not placing trades.
- Complex portfolio optimization beyond simple/robust weighting schemes (full-blown mean-variance with tight constraints can wait).

These may appear in later milestones but should not sneak into the v1 design or requirements.

### Requirements (Initial Hypotheses)

These are **hypotheses** to be validated as the pipeline solidifies. They will be refined and assigned REQ-IDs in `REQUIREMENTS.md`.

#### Validated

- The existing `legacy/` implementation and `src/market_regime/` package already:
  - Ingest key macro series (multpl + FRED) and ETF prices.
  - Engineer features via log transforms, Bernstein gap filling, and smoothed derivatives.
  - Perform PCA + clustering and regime profiling.
  - Train supervised classifiers and compute regime-conditional ETF returns.

These constitute the **baseline capabilities** that v1 will refine and productize.

#### Active (to validate in v1)

- [ ] The pipeline can be run end-to-end via a clear CLI to:
      - Ingest/refresh data.
      - Recompute features, clustering, and supervised models.
      - Generate dashboard artifacts and plots.
- [ ] Regime labels are interpretable and stable enough to support ETF-level decisions.
- [ ] For a given date/quarter, the system can:
      - Report the most likely current regime.
      - Show historical ETF performance for similar regimes.
      - Estimate expected return for the current ETF portfolio.
- [ ] Models can produce at least one-step-ahead (next quarter) predictions for:
      - Regime transitions.
      - Directional ETF and candidate-portfolio behavior.
- [ ] The system can generate a machine-readable summary (e.g. CSV/JSON + text) that can be turned into a weekly email with:
      - Current regime and confidence.
      - Notable regime-transition risks.
      - ETF-level buy/hold/sell suggestions rooted in the regime analysis.

#### Out of Scope

- Individual-stock selection or ranking; all analysis is at the ETF level.
- Handling intraday data or sub-daily decision-making.
- Auto-executing trades with brokers or exchanges.
- Direct integration with individual crypto wallets or exchanges.

### Key Decisions (So Far)

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on ETF-level portfolios only (no individual stocks) | Simpler, more robust universe; aligns with available data and reduces overfitting risk. | **Locked for v1** |
| Allow bitcoin exposure only via ETF wrappers | Avoid operational/security complexity of direct crypto while still capturing the macro signal. | **Locked for v1** |
| Weekly report cadence | Matches the regime/quarterly focus; avoids false precision from daily/intraday noise. | **Locked for v1** |

---

*Last updated: 2026-03-16 after project initialization questioning*

