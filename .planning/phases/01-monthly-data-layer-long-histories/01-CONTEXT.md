# Phase 1: Monthly Data Layer & Long Histories - Context

**Gathered:** 2026-07-09
**Status:** Ready for planning

<domain>
## Phase Boundary

The data foundation everything else stands on: monthly-frequency ingestion and
transforms, spliced core-asset histories back to ~1962, ALFRED point-in-time
vintages for revision-prone agency series, a fast/slow/agency feature taxonomy in
config, and NULL-tolerant ingestion of satellites + Glenn's holdings. Requirements
DATA-01…06. No modeling, no honesty infrastructure (Phase 2), no labeling (Phase 3).

</domain>

<decisions>
## Implementation Decisions

### Architecture (where the monthly spine lives)
- **D-01:** New parallel subpackage (e.g. `src/trading_crab_lib/platform/`) with its own
  ingestion/splicing modules and its own checkpoint namespace (`monthly_*` or a separate
  checkpoint subdir). The quarterly incumbent (9-step pipeline, 769 tests) is FROZEN and
  must not be modified. Existing fetchers (fred.py, multpl.py, macrotrends.py, assets.py)
  may be imported/reused from the new package but not edited beyond what import-reuse
  strictly requires.
- **D-02:** The new subpackage is the Phase 6 migration unit — keep its internal imports
  self-contained so it lifts cleanly into `strycker/trading-crab`.

### Splicing policy
- **D-03:** "Model the index, trade the ETF." Each asset CLASS gets ONE spliced research
  series used by all models; the weekly report maps each class to Glenn's tradable ticker.
  Class → research series → tradable mapping:
  - US equities: S&P 500 total return (Shiller/multpl history → modern index) → SPY
  - Long duration: Treasury total-return synthetic from CMT yields (GS10) → TLT
  - Gold: macrotrends monthly (1915+) / LBMA → **IAU** (Glenn holds IAU, not GLD)
  - Oil: WTI spot (macrotrends 1946+ / FRED) → oil proxy ticker (USO or leave class-level)
  - Cash: 3-month T-bill yield (TB3MS) as return series → **FZFXX/SPAXX** (Fidelity money markets)
- **D-04:** Splice joins: ratio-splice at overlap windows (scale older segment to match the
  newer at the seam); every splice rule documented per asset (source, date range, join date,
  method) in a docs file or config block — this documentation is a DATA-02 deliverable.
- **D-05:** Store daily raw prices where sources provide daily; derive the monthly spine by
  resampling. Daily persistence is deliberate — the Phase 4 tripwire consumes daily data.

### ALFRED scope
- **D-06:** True point-in-time vintages ONLY for the revision-heavy agency series used in
  labeling/features: GDP, CPI (CPIAUCSL), UNRATE, INDPRO, PAYEMS (+GNP if kept). All other
  agency-ish series keep publication-lag shift. Market-observed series (rates, spreads,
  prices) never need vintages. Pre-vintage-era fallback = publication-lag alignment,
  documented as an accepted compromise (ALFRED archives mostly start in the 1990s).
- **D-07:** ALFRED uses the same FRED_API_KEY (already in env) — no new credentials.

### Universe (tickers)
- **D-08:** Core 5 classes (equities, long duration, gold, oil, cash) via spliced research
  series per D-03.
- **D-09:** Key satellites to ingest and score (NULL-tolerant, shorter histories OK):
  QQQ, IWM, EFA, IEF, SHY, VNQ, VYM, AGG, HYG, LQD, EEM, DBA, MCD, COST, O, TSM,
  IAU, SLV, GDX.
- **D-10:** Glenn's current holdings (must always be scored): IAU, SLV, GDX, DBA, FZFXX, SPAXX.
- **D-11:** Watchlist (not held, score from day one): UNG, UEC.
- **D-12:** FZFXX/SPAXX are Fidelity money-market funds — no price ingestion (NAV ≈ $1);
  they map to the cash class whose return series is the 3-month T-bill yield.
- **D-13:** Single names in the universe (MCD, COST, O, TSM, UEC) ingest like ETFs
  (daily adjusted close via yfinance chain); no fundamental data in v1.

### Claude's Discretion
- Exact subpackage name and module layout (planner decides; follow existing conventions).
- Checkpoint naming scheme for the monthly namespace.
- Which free source wins per splice segment when several qualify (researcher investigates
  Shiller vs multpl vs Stooq coverage/quality and recommends).
- Treasury TR synthetic construction method (standard CMT-yield-to-total-return
  approximation; researcher picks the formula and documents it).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design (authoritative)
- `platform_design/platform_design.md` §9 — data specification (span, splicing, feature
  taxonomy, transforms); §11 R1/R6/R13 — the gaps this phase closes; §2 D10–D12 — locked
  decisions on frequency, data start, vintage discipline.

### Requirements & planning
- `.planning/REQUIREMENTS.md` — DATA-01…06 (this phase's requirement text).
- `.planning/ROADMAP.md` — Phase 1 goal + success criteria.
- `ROADMAP.md` (repo root) — Tier 0 T0.2/T0.3 product-level framing.

### Codebase
- `.planning/codebase/INTEGRATIONS.md` — existing fetcher inventory (FRED parallel fetch,
  multpl 2s rate limit, macrotrends monthly gold 1915+/WTI 1946+, yfinance fallback chain
  incl. Stooq), checkpoint conventions.
- `CLAUDE.md` (repo root) — conventions, pipeline-order invariants, pitfalls P1–P27
  (esp. P7 rate limits, P22 SSL, P26 FRED key).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/trading_crab_lib/ingestion/macrotrends.py`: already returns MONTHLY gold (1915+)
  and WTI (1946+) — direct splice segments for two core classes.
- `src/trading_crab_lib/ingestion/fred.py`: ThreadPool fetcher with per-series
  publication-lag shift — pattern to extend for ALFRED vintages (fredapi supports
  `get_series_as_of_date` / ALFRED endpoints with the same key).
- `src/trading_crab_lib/ingestion/assets.py`: yfinance batch + Stooq fallback chain —
  reuse for daily satellite/holdings prices.
- `src/trading_crab_lib/ingestion/multpl.py`: S&P, dividend yield, CAPE etc. — inputs to
  the S&P total-return splice and slow-layer valuation anchors.
- `src/trading_crab_lib/checkpoints.py`: CheckpointManager (parquet + manifest) — reuse
  as-is with a new checkpoint namespace.

### Established Patterns
- Config-driven series lists in `config/settings.yaml` (never hardcode URLs/series IDs).
- `from __future__ import annotations`, type hints on public functions, logging not print.
- Graceful degradation on network failure (WARNING + continue), completeness report after
  ingestion.

### Integration Points
- New subpackage imports fetchers; does NOT modify them. Incumbent pipeline steps 1–9 and
  their checkpoints stay byte-identical in behavior.
- New config section(s) in `settings.yaml` for: monthly data spine, splice definitions,
  feature taxonomy (fast/slow/agency), universe (core classes / satellites / holdings /
  watchlist), paid-provider placeholder seams.

</code_context>

<specifics>
## Specific Ideas

- Gold maps to IAU (not GLD) because that's the actual holding — same asset whose stop-loss
  gap incident motivated design §27.
- Paid-provider seams: document Norgate / Tiingo / EODHD as adapter placeholders (interface
  notes only, no implementation). Note stockcharts.com and finviz.com (Glenn has paid
  subscriptions) as candidate FEATURE sources for later milestones — not price-history
  sources for this phase.
- ALFRED pre-vintage compromise must be stated in the splice/vintage documentation, not
  silently absorbed.

</specifics>

<deferred>
## Deferred Ideas

- Fidelity positions-CSV parser (L4-V2-05) — Phase 4 uses manual YAML.
- Survivorship-clean constituent data for breadth features (DATA-V2-01) — needs paid provider.
- stockcharts/finviz feature ingestion (DATA-V2-02).

</deferred>

---

*Phase: 1-Monthly Data Layer & Long Histories*
*Context gathered: 2026-07-09*
