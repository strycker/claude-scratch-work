# Architecture Decision Records — `platform_design/adr/`

This directory holds Architecture Decision Records (ADRs) for the `platform/` codebase
(`src/trading_crab_lib/platform/`) and its supporting design document,
`platform_design/platform_design.md`. **This convention is new as of Phase 7
(2026-09-14/15) — no ADR file or directory existed in this repo before it.**

## Why a separate convention from the legacy root `CLAUDE.md` ADR log

The legacy root `CLAUDE.md` (repo root, one level up) already has a numbered ADR log
(`### ADR #1` through `### ADR #12`, "Architecture Decision Records" section). That log is
explicitly scoped to the **frozen legacy quarterly pipeline** (`legacy/unified_script.py`,
`src/trading_crab_lib/`'s pre-platform modules — clustering, transforms, prediction, etc.) — a
separate, frozen codebase that this repo keeps only as ground truth and does not extend. Its
numbering, its location (inline in `CLAUDE.md` rather than standalone files), and its narrative
prose style are conventions for that other pipeline. They do not apply to `platform/`, which is
a fresh, actively-developed codebase (Phase 6 D-01: `platform/` imports nothing from the legacy
library) with its own decisions and its own audit trail (`.planning/UAT-AUDIT-2026-09-09.md`,
the phase `CONTEXT.md`/`MEASUREMENTS.md` documents).

Rather than appending `platform/` decisions to the legacy log — which would conflate two
codebases' histories under one numbering scheme — this directory establishes a **separate,
standalone ADR sequence** for `platform/` and its design document. `.planning/` phase
directories remain the process/planning record (what was executed, when, by whom); this
directory is the **architecture record** (what was decided, and why), living beside the design
document it amends.

## Convention

- **One file per decision.** No decision is folded into another file's "Amendments" section
  once it becomes clear it deserves its own record.
- **Filename:** zero-padded four-digit sequence prefix + kebab-case slug —
  `NNNN-kebab-case-slug.md` (e.g. `0001-l1-feature-policy.md`). The sequence number is
  monotonically increasing across the whole `platform/` codebase, never reused.
- **Canonical section headers**, in this order (matching the vocabulary
  `.claude/gsd-core/bin/lib/adr-parser.cjs` recognizes, so GSD tooling can parse ADRs in this
  directory without special-casing them):
  - `## Status` — accepted / superseded / rejected / deprecated, dated, naming the phase and
    requirement IDs the decision serves.
  - `## Context` — the problem, its diagnosed mechanism, and why it is a decision rather than a
    bug fix (where applicable).
  - `## Decision` — what was decided, stated as a concrete, checkable claim.
  - `## Considered Options` — every alternative seriously considered, each with its rejection
    reason. An alternative rejected on the merits should be argued; an alternative rejected
    because it was tried and measured should cite the measurement (a logged trial, a registry
    row, a concrete number) — never rejected by assertion alone when evidence was available.
  - `## Consequences` — what changes as a result, including costs and what becomes re-dated or
    invalidated.
  - Additional sections beyond these five are permitted and expected (e.g. a `Trial ceiling`
    section, a `Deferrals and open items` section) — the five above are the required minimum,
    not a ceiling on structure.
- **Never delete a superseded ADR.** Mark it `## Status: superseded by NNNN-slug.md` in place
  and leave the rest of the file intact — the historical record of what was believed and why is
  as valuable as the current decision.
- **Cross-reference, don't duplicate.** `platform_design.md` gets a single cross-reference line
  pointing at the ADR that amends a given section; it does not restate the ADR's content
  inline. The ADR is the source of truth for the decision; the design document points at it.

## Index

| # | Slug | Decision | Status |
|---|------|----------|--------|
| 0001 | `l1-feature-policy` | Freeze the walk-forward driver's L1 labeler to the same ten-column feature space the evaluation's smoothed reference already computes (resolves audit item A13) | Accepted, 2026-09-15 |
