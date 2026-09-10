---
phase: 6
slug: platform-notebook-suite
status: in_progress
created: 2026-09-10
tests_total: 2
tests_passed: 1
tests_pending: 1
---

# Phase 6 — UAT

Two items were routed to human verification by `06-VERIFICATION.md`. One was
mechanically settleable and has been settled; one requires human judgement.

## UAT-1 — Criterion 1: all six notebooks run top-to-bottom, fresh ✅ PASSED

**Why it was routed to a human:** D-17 deliberately rejected execution-based CI
(no nbmake/papermill). The committed notebooks proved *one* successful run had
happened in an executor's environment — not that a fresh run reproduces it. That is
an existence-shaped evidence gap, exactly the class the Phases 1–5 audit warned about.

**How it was closed (2026-09-10):** each notebook was copied to a temp dir, **all
outputs and execution counts cleared**, and executed from scratch via `nbclient`
against the real checkpoints with cwd at the repo root. Committed notebooks were
never overwritten.

| Notebook | Result | Time |
|---|---|---|
| P1_data_spine | OK | 35.7s |
| P2_features_taxonomy | OK | 6.6s |
| P3_regime_labeling | OK | 14.1s |
| P4_nowcaster | OK | 5.3s |
| P5_assets_allocation | OK | 8.4s |
| P6_backtest_evaluation | OK | 5.7s |

**6/6 executed cleanly from cleared state.** Zero exceptions.

**Isolation proven live, not just tested:** `git status --porcelain` was completely
empty afterward — no writes to `data/checkpoints/platform/`, `outputs/reports/platform/`,
or `registry/`. `label_regimes()` wrote `regime_labels.parquet`,
`regime_confidences.parquet`, and `regime_profiles.parquet` into the gitignored scratch
twin `data/checkpoints/platform_notebook/`, exactly as D-10 / threat T-06-12 intended.

This upgrades criterion 1's evidence from *"a committed artifact shows a run happened"*
to *"a fresh run today reproduces it."* Note the standing limit: this is a one-off
manual check, not a standing CI gate — D-17 chose that deliberately.

## UAT-2 — Criterion 2/4: the P3 cold-start sign-off ⏳ AWAITING OPERATOR

The verdict is inherently human (D-15/D-16); no agent may record it. The cell is blank
by design. Evidence the operator is judging is reproduced below.

### Regime × era contingency (row-normalised: share of each era's months per state)

|  | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 1973 oil shock | 0.00 | **1.00** | 0.00 | 0.00 | 0.00 |
| Volcker disinflation | 0.00 | **0.63** | 0.00 | 0.00 | 0.37 |
| 1987 crash | 0.00 | 0.00 | 0.00 | 0.00 | **1.00** |
| LTCM / Russia default | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 |
| dot-com bust | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 |
| global financial crisis | **0.58** | 0.00 | 0.00 | 0.42 | 0.00 |
| NBER recession | 0.13 | 0.28 | 0.22 | 0.20 | 0.16 |
| *all months (baseline)* | *0.016* | *0.140* | *0.319* | *0.406* | *0.119* |

### Column-normalised (share of each state's months coming from each era)

State 0's months are **100% global-financial-crisis** months. It is a GFC detector,
not a general crisis detector — it did not fire in 1973, 1987, dot-com, or LTCM.

### Occupancy (reference labeling, 695 months)

1.58% / 13.96% / 31.94% / 40.58% / 11.94% — sums to 1.000000000000.
State 0 at 1.58% trips the §4.4 5% soft floor; report-only, not blocking (D-02).
Pooled median sojourn 97.0 months.

### Reads that cut in different directions

- **For:** every named crisis lands overwhelmingly in a *single* state, and different
  crises land in *different* states. 1973 → state 1 at 100% vs a 14% baseline.
  GFC → state 0 at 58% vs a 1.6% baseline (~37× lift).
- **Against:** NBER recessions spread almost like the unconditional baseline
  (0.13/0.28/0.22/0.20/0.16). The regimes track **crises**, not **recessions**.
- **Caveat:** state 0 holds ~11 of 695 months. Its 100%-GFC purity is a statement
  about very few observations.

### Also visible in P3, and relevant to the verdict

- A13: reference vs filtered disagree on **389 / 470 = 82.8%** of compared months,
  with no diagonal structure. The §5.4 ratio stays labelled not-interpretable.
- Coverage: 470 of 588 walk-forward steps (79.9%); the filtered path starts 1974-02.
- P3 computes its own labeling over 372 months (occupancy 2.4/15.6/32.3/32.0/17.7)
  *in addition to* loading the 695-month reference. The contingency table above uses
  the **reference**. Worth knowing which object the sign-off is about.

**To record the verdict:** open `notebooks/platform/P3_regime_labeling.ipynb`, fill the
final markdown cell's Date / Verdict / Reasoning. Per D-16 a negative verdict is valid
and does not block the phase.
