# Phase 4: Asset Prediction & Allocation - Context

**Gathered:** 2026-07-22
**Status:** Ready for planning

<domain>
## Phase Boundary

L3 asset prediction (returns-by-regime tables, EWMA vol forecasts) and L4 allocation &
report (naive vol-targeted regime-tilt with hysteresis, weekly report with trades
implied vs manual YAML holdings, 3-signal daily tripwire). Requirements L3-01/02,
L4-01..04. No honest backtest (Phase 5), no migration (Phase 6). v1 allocation is
deliberately naive — BL/HRP, Kelly, stops, and the crash dashboard are all v2.

</domain>

<decisions>
## Implementation Decisions

### Holdings YAML schema (L4-03)
- **D-01:** **Weights (fractions) per account.** One YAML per account (e.g.
  `config/accounts/<account>.yaml`, gitignored like other `*.local.yaml` secrets-adjacent
  files; a committed `config/accounts/example.yaml` documents the schema). Each file maps
  ticker → weight fraction; weights + cash must sum to ~1.0 (validate with tolerance,
  warn don't fail). No price lookups needed — the report compares target weights vs
  current weights directly. Fidelity CSV parser documented as a placeholder seam only.

### Weekly report delivery (L4-02)
- **D-02:** **Always write `weekly_report` markdown; email is opt-in** behind the
  existing incumbent pattern (`--send-email` + SMTP config / TC_* env vars). Reuse the
  incumbent's `email.py` machinery (markdown→HTML, multipart) — do not fork it; the
  platform report writer produces the markdown, the existing sender delivers it.
  Platform report artifact lives under `outputs/reports/platform/` to avoid colliding
  with the incumbent's weekly_report.md.

### Portfolio target volatility (L4-01)
- **D-03:** **10% annualized** target portfolio vol, exposed as a config knob
  (`allocation.target_vol_annual: 0.10`). Defensive by design — the core value is
  beating SPY net of avoided drawdowns, not matching its risk. Position scale ∝
  target_vol / σ̂_portfolio, long-only, capped at 1.0 (no leverage), cash absorbs the
  residual (FZFXX/SPAXX are the cash sleeve).

### Tripwire v1 signal set (L4-04)
- **D-04:** **Confirmed roadmap trio**, one per independent family:
  1. Realized-vol spike (vol family) — short-window realized vol vs its trailing baseline
  2. Credit-spread velocity (credit family) — BAA–AAA widening rate
  3. Drawdown-from-peak on SPY (price family)
  OR-logic escalation: none → "run weekly scoring early" (1 trigger) → "Tier-1 de-risk
  review" (2+ triggers). All thresholds config-driven (`tripwire:` section). Daily
  cadence, manual CLI run in v1.

### Carried forward (locked earlier — do not re-decide)
- Hysteresis bands: act when regime P crosses ~0.7, unwind below ~0.4, hold in between
  (design §5.3; ROADMAP criterion 3). Expose both in config.
- v1 asset universe: Core 5 + satellites + holdings — SPY, GLD, TLT, USO + QQQ, IWM,
  EFA, IEF, SHY, VNQ, VYM, AGG, HYG, LQD, EEM, DBA, MCD, COST, O, TSM, IAU, SLV, GDX,
  FZFXX, SPAXX (cash). Availability limited by what Phase 1 ingested; NULL-tolerant for
  short histories (D11).
- EWMA vol per asset (design §6.2): position size ∝ 1/σ̂; realized-vol features also
  feed the tripwire. GARCH/DCC/Ledoit–Wolf covariance → v2 (L3-V2-01).
- Return forecasts shrunk brutally (design §7): the regime tilt comes from historical
  regime-conditional stats (Phase 3 labels + returns-by-regime tables), never from raw
  point-forecast optimization.
- Nowcaster probabilities (Phase 3 `platform/prediction/nowcaster.py`) drive the regime
  distribution; empirical transition matrix gives the trajectory diagnostic.
- All new code in `src/trading_crab_lib/platform/` (e.g. `platform/assets/`,
  `platform/allocation/`, `platform/report/` — planner's call); incumbent untouched
  except read-only reuse of `email.py` senders.
- Honesty rails apply: dev data ≤2020-12; any evaluated configuration → registry;
  causal features only in anything supervised (returns-by-regime tables are historical
  conditioning, not supervised prediction — no CV needed in v1).
- v2 deferrals bound this phase: L3-V2-01/02/03 (covariance, MoE, fair-value gap),
  L4-V2-01..04 (BL/HRP/Kelly, stops, crash dashboard, tactical sleeve), full tripwire
  orchestrator (L2-V2-03).

### Claude's Discretion
- EWMA half-life/lambda convention for monthly (and daily tripwire) vol — literature
  default (e.g. RiskMetrics-style), exposed in config.
- Regime-tilt construction details: how regime-conditional stats map to tilt weights
  (e.g. within-regime Sharpe rank, clipped ≥ 0, blended by nowcaster probabilities),
  provided it is naive, transparent, and documented in the report.
- Hysteresis state persistence (where the "last acted regime" state lives between runs).
- Report layout/sections, and module split inside `platform/`.
- Tripwire threshold defaults (documented derivation, config-overridable).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design (authoritative)
- `platform_design/platform_design.md` §6.2 (EWMA vol + vol targeting doctrine), §6.5
  (prediction targets — excess/relative returns framing), §7 (L4 doctrine: shrink
  forecasts, weekly report output list), §5.3 (hysteresis anti-flicker), §23.2 + §25
  (tripwire exit-side + multi-timescale consensus — v1 takes the minimal 3-signal OR
  subset), §16.7 (long-only, no options/shorts/crypto/MLPs; FZFXX/SPAXX cash), §14
  Phase 1 tracer bullet (naive allocation is the point), D11 (NULL-tolerant short
  histories).

### Requirements & planning
- `.planning/REQUIREMENTS.md` — L3-01/02, L4-01..04; v2 deferrals L3-V2-*, L4-V2-*.
- `.planning/ROADMAP.md` — Phase 4 goal + 5 success criteria.
- `.planning/phases/03-regime-labeling-prediction/03-CONTEXT.md` +
  `03-0{1..4}-SUMMARY.md` — the L1/L2 APIs this phase consumes.
- `.planning/phases/02-honesty-infrastructure/02-CONTEXT.md` — honesty rails.

### Codebase
- `src/trading_crab_lib/platform/labeling/{jump_model,diagnostics}.py` — labels,
  confidences, profiles (label_regimes outputs).
- `src/trading_crab_lib/platform/prediction/{nowcaster,transition_matrix}.py` — regime
  distribution + trajectory inputs to the report.
- `src/trading_crab_lib/platform/checkpoints.py`, `config.py` — persistence + config.
- `src/trading_crab_lib/platform/ingestion/prices_daily.py` + splice module — daily
  spliced price histories for vol/tripwire/returns.
- `src/trading_crab_lib/email.py` (incumbent) — reused sender (read-only; do not
  modify).
- `config/platform_settings.yaml` — gains `allocation:`, `tripwire:`, `report:`
  sections (defensive .get() reads, Phase 2 pattern).
- `CLAUDE.md` (root) — conventions.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 3's `label_regimes()` checkpoint outputs (labels/confidences/profiles) and
  `fit_nowcaster`/`predict_proba` — the regime inputs to everything here.
- `platform/honesty/gap_lag.py` module shape (compute_* + report_* + __main__
  self-check + parquet artifact) — the analog for returns-by-regime, vol, tripwire
  modules.
- Incumbent `email.py` — send path; incumbent `reporting.py` — layout inspiration only
  (do not import into platform).
- pandas `.ewm()` covers EWMA vol — zero new dependencies expected.

### Established Patterns
- Config sections read via `.get()`, never added to `_REQUIRED_PLATFORM_SECTIONS`.
- Synthetic-frame no-network tests; invariant-style headline tests (e.g. hysteresis:
  a probability path oscillating 0.65↔0.72 must NOT flip the mix each month; tripwire:
  exactly-one-trigger → early-scoring, two → Tier-1).
- TDD RED→GREEN commits for correctness-critical logic (hysteresis state machine,
  vol targeting math, tripwire OR-logic).

### Integration Points
- Consumes: monthly_features / labels / nowcaster probabilities / daily spliced prices.
- Produces: weekly report markdown + parquet artifacts under
  `outputs/reports/platform/`; tripwire CLI exit output.
- Nothing modifies the incumbent pipeline; email sender reused as-is.

</code_context>

<specifics>
## Specific Ideas

- The report is the product surface Glenn reads before trading in Fidelity — trades
  implied must be explicit and unambiguous (per account: BUY X to +Δw, SELL Y to −Δw,
  HOLD Z), with the regime rationale one line each.
- Hysteresis must be tested as a state machine with a persisted state artifact — the
  no-flip invariant is the headline test of L4-01.
- Tripwire output is a single escalation enum printed by a CLI entry point
  (`python3 -m ...tripwire`) so it can be run daily by hand (automation is a v2
  placeholder, OPS-V2-01).

</specifics>

<deferred>
## Deferred Ideas

- BL/HRP weights, fractional Kelly, no-trade band math (§21) — v2 (L4-V2-01).
- Model-driven vol-scaled stops + §27 policy stack — v2 (L4-V2-02).
- Crash-probability dashboard + crisis-type conditioning — v2 (L4-V2-03).
- Regime-conditional covariance, Ledoit–Wolf, GARCH/DCC — v2 (L3-V2-01).
- MoE + boosted ceiling model; fair-value gap module — v2 (L3-V2-02/03).
- Full tripwire orchestrator with family-independence voting (§25) — v2 (L2-V2-03).
- Fidelity positions-CSV parser — v2 (L4-V2-05); seam documented in v1.
- Automated scheduled runs + email delivery — v2 (OPS-V2-01).
- Tactical sleeve reporting (§16.5) — v2 (L4-V2-04).

</deferred>

---

*Phase: 4-Asset Prediction & Allocation*
*Context gathered: 2026-07-22*
