# Archived Planning Documents

Historical planning documents, kept for provenance. **None of these describe current
work.** They were moved here on 2026-08-04 because they were either self-marked
SUPERSEDED, fully executed, or describe the legacy quarterly pipeline rather than the
L0–L4 platform.

For current state, see:

| Question | Authoritative source |
|---|---|
| What is the project's design? | `platform_design/platform_design.md` (v1.7) |
| What's the plan / where are we? | `.planning/ROADMAP.md`, `.planning/STATE.md` |
| What's done and what's next? | `.planning/STATUS-REVIEW-2026-08.md` |
| How do I migrate to the public repo? | `MIGRATION-PLAN.md` (repo root) |
| How do I work in this repo? | `CLAUDE.md` (repo root) |

---

## What's here and why

| File | Why archived |
|---|---|
| `META_PLAN.md` | Self-marked ⚠️ SUPERSEDED (July 2026) |
| `NEXT_STEPS.md` | Self-marked ⚠️ SUPERSEDED (July 2026) |
| `PHASE_B_PLAN.md` | Self-marked ⚠️ SUPERSEDED; work executed (see CLAUDE.md D30–D39) |
| `PHASE_C_PLAN.md` | Self-marked ⚠️ SUPERSEDED; work executed (D23–D29) |
| `PHASE_D_PLAN.md` | Self-marked ⚠️ SUPERSEDED; work executed (D30–D38) |
| `PHASE_E_PLAN.md` | Self-marked ⚠️ SUPERSEDED; work executed (D39) |
| `MONITORING_EXPANSION_PLAN.md` | Fully executed — all phases A–E complete (D39) |
| `RENAME_PLAN.md` | Fully executed — `market_regime` → `trading_crab_lib` (D15) |
| `REBUILD-FROM-SCRATCH-GUIDE.md` | Describes rebuilding the **legacy quarterly pipeline**, superseded by the platform |
| `STATE.md` | Root-level duplicate; `.planning/STATE.md` is the live GSD-managed state file |

## Note on the legacy quarterly pipeline

Several of these describe the 9-step quarterly pipeline (`pipelines/01–09`,
`notebooks/01–12`, the non-`platform/` modules in `src/trading_crab_lib/`). That code
still runs and still passes its tests, but it is **superseded by the L0–L4 platform**
and is not the migration target. It stays in this repo as reference, in the same spirit
as `legacy/unified_script.py`.
