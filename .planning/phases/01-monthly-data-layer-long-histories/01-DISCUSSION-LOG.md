# Phase 1: Monthly Data Layer & Long Histories - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-09
**Phase:** 1-Monthly Data Layer & Long Histories
**Areas discussed:** New-code architecture, Splicing policy, ALFRED scope, Universe tickers

---

## New-code architecture

| Option | Description | Selected |
|--------|-------------|----------|
| New parallel subpackage | Own ingestion/splicing/checkpoint namespace; incumbent untouched; clean Phase 6 migration unit | ✓ |
| Frequency switch in existing modules | Thread monthly/quarterly param through existing code; risks frozen incumbent's 769 tests | |
| You decide | Claude picks during planning | |

**User's choice:** New parallel subpackage (Recommended)

---

## Splicing policy

| Option | Description | Selected |
|--------|-------------|----------|
| Model index, trade ETF | One spliced research series per asset CLASS; report maps to tradable ticker | ✓ |
| Backfill ETF price series | Extend each ETF's own history backward with scaled index data | |
| You decide | Claude picks during planning | |

**User's choice:** Model index, trade ETF (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Daily raw → monthly spine | Persist daily where available; resample to monthly; tripwire-ready | ✓ |
| Monthly only for now | Smaller/simpler; tripwire adds daily later | |
| You decide | Claude picks during planning | |

**User's choice:** Daily raw → monthly spine (Recommended)

---

## ALFRED scope

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal revision-prone set | Vintages for GDP, CPI, UNRATE, INDPRO, PAYEMS (+GNP); rest keep lag-shift | ✓ |
| All agency series | Vintage-fetch every FRED series with an ALFRED archive | |
| You decide | Claude picks during planning | |

**User's choice:** Minimal revision-prone set (Recommended)

---

## Universe tickers

Freeform (no options — user-provided lists):

**Satellites:** QQQ, IWM, EFA, IEF, SHY, VNQ, VYM, AGG, HYG, LQD, EEM, DBA, MCD, COST, O, TSM, IAU, SLV, GDX, FZFXX, SPAXX
**Holdings:** IAU, SLV, GDX, DBA, FZFXX, SPAXX
**Watchlist (not held):** UNG, UEC

**Notes:** FZFXX/SPAXX are Fidelity money markets → mapped to cash class (no price series;
3-month T-bill yield is the cash return proxy). IAU (not GLD) is the gold-class tradable
mapping — it's the actual holding. Single names (MCD, COST, O, TSM, UEC) ingest like ETFs.

---

## Claude's Discretion

- Exact subpackage name and module layout
- Checkpoint naming scheme for the monthly namespace
- Per-segment splice source selection (researcher recommends)
- Treasury total-return synthetic construction formula

## Deferred Ideas

- Fidelity positions-CSV parser (Phase 4 uses manual YAML; v2 backlog L4-V2-05)
- Survivorship-clean constituent data / paid providers (DATA-V2-01)
- stockcharts.com / finviz.com feature ingestion (DATA-V2-02)
