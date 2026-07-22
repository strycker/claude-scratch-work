# Phase 2: Honesty Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-16
**Phase:** 2-Honesty Infrastructure
**Areas discussed:** Trial registry storage, Holdout carve mechanics, Gap/lag reporting surface

---

## Trial registry storage

| Option | Description | Selected |
|--------|-------------|----------|
| Append-only JSONL committed to git | Immutable pre-registration ledger; survives machine loss; tamper-evident history | ✓ |
| SQLite in data/ (gitignored) | Better ad-hoc querying; local artifact, no audit history | |
| Hybrid | JSONL ledger of record + rebuildable SQLite view | |

**User's choice:** (a) JSONL committed to git

---

## Holdout carve mechanics

| Option | Description | Selected |
|--------|-------------|----------|
| Ingestion-level split | Dev checkpoints end at 2020-12; 2021+ rows only in data/holdout/; strongest wall | ✓ |
| Loader-level guard | Full-history files, default loaders truncate unless live_mode=True; weaker | |

**User's choice:** (a) Ingestion-level split

---

## Gap/lag reporting surface

| Option | Description | Selected |
|--------|-------------|----------|
| CLI + persisted artifact now; weekly report in Phase 4 | Phase 2 stays pure infrastructure | ✓ |
| Also bolt onto incumbent weekly report now | Earlier visibility; touches frozen incumbent | |

**User's choice:** (a) CLI + artifact now, report wiring in Phase 4

---

## Claude's Discretion

- Purged/embargoed CV parameter conventions (López de Prado ch. 7), exposed in config
- Registry JSONL schema and platform/ module layout
- Trivial model definition for the walk-forward end-to-end proof

## Deferred Ideas

- Weekly-report wiring of gap/lag metrics (Phase 4)
- DSR computation against the registry (design freeze, later milestone)

(Note: AskUserQuestion UI failed twice this session; areas and options were presented as
plain-text numbered lists and answered "1a, 2a, 3a".)
