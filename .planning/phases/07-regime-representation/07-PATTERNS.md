# Phase 7 (Wave 1 only): Regime Representation - Pattern Map

**Mapped:** 2026-09-14
**Scope:** Wave 1 deliverables only (feature-policy unification, criteria 1-4, ADR). Wave 2
(leadership classifier, relative-strength features, INV-01) is out of scope for this pass —
see the "Wave 2 — not this pass" appendix at the end for the few observations worth carrying
forward, per the orchestrator's instruction to keep that section minimal.

**Files analyzed:** 9 (5 modified library files, 1 new script, 3 test files touched/extended)
**Analogs found:** all in-repo — this phase is a restructuring of existing correct functions,
not new algorithm work, so every "analog" is the function being modified itself or its
immediate sibling in the same module.

---

## File Classification

| File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `platform/backtest/driver.py::_refit_l1` | labeling (L1 fit, per-step) | batch/CRUD-like (fit-predict) | `evaluation/report.py::_reference_label_columns` (the function it must now call) | exact — same policy, different caller |
| `platform/evaluation/report.py::run_full_backtest_evaluation` | orchestration (evaluation) | batch | itself (reordering only — `first_decision` computed earlier) | exact — self-analog |
| `platform/evaluation/sojourn_lag.py` | evaluation metric | transform | unchanged — pattern source for "already does the right thing" | n/a (no changes needed) |
| `platform/plotting/core.py::A13_CAVEAT` | reporting/plotting constant | transform | itself (string content edit only) | exact |
| new: `scripts/recompute_monthly_features.py` (or similarly named) | script / one-off CLI | batch, offline recompute | `scripts/build_platform_data.py` (structure/CLI shape) + `honesty/holdout.py::write_monthly_features_split` (the actual write call) | role-match — `build_platform_data.py` is the sibling script but does full ingest, NOT what this needs |
| `tests/unit/test_platform_backtest_driver.py` (new equivalence test) | test (unit, equivalence) | request-response (assert-only) | `tests/unit/test_platform_evaluation_report.py::TestReferenceLabelColumns` (lines 334-366) | exact — same function under test, extend to compare against driver output |
| `tests/unit/test_platform_evaluation_report.py` | test (unit) | request-response | itself — `TestReferenceLabelColumns`, `TestSmoothedHindsightUniverse` | exact |
| `tests/unit/test_platform_evaluation_sojourn_lag.py` | test (unit) | request-response | itself — extend for `n_resolved`/`n_transitions` assertions already present in the source | exact |
| `tests/unit/test_platform_plotting_regime.py:206` (`test_real_dev_features_reproduce_the_seven_a13_change_points`) | test (golden/pinned regression) | request-response | itself — re-pin `EXPECTED_CHANGE_POINTS` (lines 36-43) | exact — no OTHER re-pinning precedent exists in the repo (see finding #3 below) |
| ADR document (new) | documentation | n/a | **no analog found** — see finding #5 | n/a |

---

## Pattern Assignments

### 1. The equivalence test (criterion 1) — closest cousin found, quoted

**No prior "two computations must produce the same list" test exists in this repo.** The
closest and *intended* cousin — per RESEARCH.md's own "Don't Hand-Roll" table and Code
Examples section — is `TestReferenceLabelColumns` in
`tests/unit/test_platform_evaluation_report.py:334-366`, which tests
`_reference_label_columns` **in isolation** today. It is not yet an equivalence test; it is
the *foundation* the new equivalence test extends, because the shared-call-site design (D-01/
Pattern 1) means the driver's `_refit_l1` should stop having its own logic and instead consume
this same function's output.

Quoted verbatim (`tests/unit/test_platform_evaluation_report.py:334-353`):

```python
class TestReferenceLabelColumns:
    """The full-sample smoothed reference must span EVERY decision date (the
    walk-forward now labels pre-1990 under approach ii), so it keeps long-history
    columns and drops structural late-starts."""

    def test_drops_late_start_keeps_warmup_and_complete(self):
        idx = pd.date_range("1962-01-31", periods=240, freq="ME")
        first_decision = idx[120]  # ~1972, like min_train=120
        df = pd.DataFrame(index=idx)
        df["complete"] = np.arange(240, dtype=float)
        df["warmup_only"] = np.arange(240, dtype=float)
        df.iloc[:3, df.columns.get_loc("warmup_only")] = np.nan       # NaN only pre-1962Q1 (< first_decision)
        df["late_start"] = np.arange(240, dtype=float)
        df.iloc[:180, df.columns.get_loc("late_start")] = np.nan       # NaN through ~1977 (> first_decision)

        ref = report._reference_label_columns(df, list(df.columns), first_decision)

        assert "complete" in ref        # present across decision range
        assert "warmup_only" in ref     # NaN only before the first decision → kept
        assert "late_start" not in ref  # NaN within the decision range → dropped
```

**How the new equivalence test should be shaped to match house style:**

- **Location:** `tests/unit/test_platform_backtest_driver.py` (per VALIDATION.md's own
  automated command: `pytest tests/unit/test_platform_backtest_driver.py -k equivalence -x`).
- **Form — sample a few decision dates, not parametrize over all 588.** House style for
  "spans a walk-forward" assertions in this codebase samples rather than exhaustively
  parametrizes when the check is O(1) per date and the underlying property is structural (see
  `test_platform_plotting_regime.py`'s own `active_feature_count_timeline` tests, which check
  a handful of specific index positions — `idx[29]`, `idx[30]` — not every row). Sample the
  first decision date, a mid-range date, and the last dev-window date.
- **Real-checkpoint-dependent variant: skip via `pytest.mark.skipif` exactly like
  `REAL_MONTHLY_FEATURES`.** Quoted pattern from `tests/unit/test_platform_plotting_regime.py:29-32,202-206`:

```python
REAL_MONTHLY_FEATURES = Path("data/checkpoints/platform/monthly_features.parquet")
...
@pytest.mark.skipif(
    not REAL_MONTHLY_FEATURES.exists(),
    reason="real platform monthly_features checkpoint not present",
)
def test_real_dev_features_reproduce_the_seven_a13_change_points(self):
    ...
```
  The criterion-1 equivalence test should have **two variants**: (a) a synthetic-DataFrame unit
  test (no skip, always runs, mirrors `TestReferenceLabelColumns`'s synthetic-index style) that
  proves the driver's `_refit_l1(frozen_features=...)` uses EXACTLY the list passed in, filtered
  only for column presence; and (b) an integration-shaped test against the real checkpoint,
  gated by the SAME `REAL_MONTHLY_FEATURES.exists()` skip idiom, asserting
  `_reference_label_columns(...)` and the driver's active-column list are byte-identical at the
  sampled dates. **This mirrors the existing two-tier pattern already in the repo** (synthetic
  unit test + skippable real-checkpoint test) rather than inventing a third shape.
- **What the test must assert is NOT true before the fix:** a companion `xfail`/inverse
  assertion is not needed — the fix removes the second computation entirely, so there is
  nothing left to diverge; the test simply must fail today (before the fix) because
  `_refit_l1` currently has no `frozen_features` parameter at all.

---

### 2. Threading a frozen column list through the walk-forward loop — is there prior art?

**No prior art for a `frozen_features`-shaped parameter exists anywhere in `platform/`.** This
is genuinely new, but the *shape* of the addition matches an already-established convention:
**an explicit, keyword-only `X | None = None` parameter with a documented fallback**, exactly
like `run_backtest`'s own `min_train: int | None = None` (`driver.py:274`) and
`registry_path: Any = None` (`driver.py:277`). Quote from the existing signature
(`driver.py:269-278`):

```python
def run_backtest(
    monthly_features: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    min_train: int | None = None,
    cash_returns: pd.Series | None = None,
    use_regime_tilt: bool = True,
    registry_path: Any = None,
) -> tuple[pd.DataFrame, dict[str, list]]:
```

RESEARCH.md's own Pattern 1 sketch (which this file independently confirms is grounded in the
real code, not invented) proposes exactly this shape for `_refit_l1` and `run_backtest`:

```python
# driver.py::_refit_l1, new signature (RESEARCH.md Pattern 1, confirmed against real code):
def _refit_l1(train_features, cfg, *, frozen_features: list[str] | None = None) -> pd.Series:
    ...
    if frozen_features is not None:
        active = [c for c in frozen_features if c in train_features.columns]
    else:
        active = _window_active_features(train_features, lean_cols, min_history=min_history)
    ...
```

This is **not config-threading** (no new `cfg["backtest"]["frozen_features"]` key is
appropriate — the frozen list is a *derived, run-specific* value, not a tunable setting) and
**not a `cfg` sub-dict** — it is a plain keyword argument, matching every other per-run
derived value `run_backtest` already accepts (`cash_returns`, `min_train`). The planner should
NOT invent a config key for this; the existing convention is "pass it as an explicit kwarg with
a `None` default meaning 'compute it the old way,'" already used three times in this exact
function signature.

**Call-chain consequence to plan explicitly:** `run_backtest` itself needs a new keyword
(RESEARCH.md's sketch names it `frozen_l1_features`) that it threads into every `_refit_l1`
call inside the per-step loop (`driver.py:398` and, per F5's ablation-path debug branch,
`driver.py:386`) — both call sites must receive the same parameter, or the ablation leg's
(discarded) L1 fit would silently diverge from the strategy leg's.

---

### 3. Re-pinning a deliberately falsified regression test — no precedent found

**No precedent exists in this repo for re-pinning a golden/expected-value constant with a
comment naming the decision that changed it.** Searched `.planning/` and `tests/` broadly;
the only prior "these numbers moved and are recorded" mechanism found is `07-CONTEXT.md`'s own
**D-05 pre/post table** convention (a NEW artifact this same phase introduces, not a pattern
that predates it) and the legacy `CLAUDE.md`'s numbered "Development Decisions Log" (D1-D50),
which documents *why* something changed but does not re-pin a Python test constant.

**The planner must establish the convention, not match one.** Recommended shape, grounded in
this repo's existing commenting style (`# ── Section ──` dividers, docstrings explaining *why*
a non-obvious choice was made — per `.claude/CLAUDE.md` "Comments" conventions):

```python
# Re-pinned 2026-09-14 (D-02-A amendment, 07-CONTEXT.md): monthly_features was
# recomputed from the current monthly_raw checkpoint after discovering `oil`'s
# 1985-02 start in features was a staleness artifact (monthly_raw has full
# 1962-2020 coverage). The sequence below reflects the TEN-column frozen set,
# not the superseded nine-column set — see the ADR for the full history.
EXPECTED_CHANGE_POINTS = [
    ("1972-01-31", 5),
    ("1972-02-29", 7),
    ...
]
```

The exact current constant to replace (`tests/unit/test_platform_plotting_regime.py:36-43`,
read this session):

```python
EXPECTED_CHANGE_POINTS = [
    ("1972-01-31", 4),
    ("1972-02-29", 6),
    ("1972-04-30", 8),
    # ... (4 more lines not shown in this excerpt — file has 7 total entries)
]
```
This must become the `5→7→9→10→11→12→13` sequence per D-02-A, in the **same commit** that
recomputes the checkpoint (CONTEXT.md's explicit instruction) — not loosened to an inequality.

---

### 4. Offline checkpoint recompute — exact call sequence, and no existing entry point

**`scripts/build_platform_data.py` does NOT do this** — confirmed by reading its module
docstring and `main()` (read this session): it always calls `build_monthly_spine()`, which
re-fetches from FRED, multpl-equivalent scraping, macrotrends, and yfinance
(`transforms_monthly.py::build_monthly_spine`, line ~303-330). There is no `--recompute`-only
flag; RESEARCH.md's Pitfall 1 independently confirms this ("the platform has no
`--recompute`-only mode ... confirmed by reading its `main()` — no argparse flags for selective
steps"). **A new entry point is needed.**

**Exact call sequence a recompute-only script must use** (every function verified to exist at
the cited location this session):

```python
from __future__ import annotations

import logging

from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.honesty.holdout import (
    DEFAULT_HOLDOUT_CUTOFF,
    write_monthly_features_split,
)
from trading_crab_lib.platform.transforms_monthly import compute_lean_features

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    cfg = load_platform_config()
    cm = get_platform_checkpoint_manager()

    monthly_raw = cm.load("monthly_raw")          # cached — no network
    features = compute_lean_features(monthly_raw, cfg)   # pure function (transforms_monthly.py:227)

    # Re-carve at the holdout boundary and write BOTH dev + holdout sides,
    # exactly as write_monthly_features_split already does (honesty/holdout.py:57-72):
    write_monthly_features_split(features, name="monthly_features", cutoff=DEFAULT_HOLDOUT_CUTOFF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Notes grounding each call:
- `get_platform_checkpoint_manager()` — `platform/checkpoints.py:26-28`, returns a
  `CheckpointManager` scoped to `data/checkpoints/platform/`; `.load("monthly_raw")` and
  `.save(...)` (via `write_monthly_features_split`) are the incumbent
  `trading_crab_lib.checkpoints.CheckpointManager` methods, reused verbatim per that module's
  own docstring ("D-01: never subclass or reimplement save/load/is_fresh").
- `compute_lean_features(monthly_raw, cfg)` — `transforms_monthly.py:227-282`, confirmed a pure
  function with no I/O side effects (builds a `dict[str, pd.Series]` and
  `pd.concat(features, axis=1)`).
- `write_monthly_features_split(df, name, cutoff)` — `honesty/holdout.py:57-72` (read this
  session), does exactly: `split_by_holdout_boundary` then
  `get_platform_checkpoint_manager().save(dev_df, name)` +
  `get_holdout_checkpoint_manager().save(holdout_df, name)`. This is the SAME function
  `build_monthly_spine()` itself presumably calls at the end of a full rebuild (not verified
  this session past line 330 of `transforms_monthly.py` — the planner should confirm the tail
  of `build_monthly_spine()` calls this same function, to be sure the recompute script produces
  a checkpoint indistinguishable in shape from a full rebuild's output).
- After writing, `assert_dev_checkpoint_within_boundary("monthly_features")`
  (`honesty/holdout.py:108-128`) is available as a self-check the script should call before
  exiting — it raises `RuntimeError` if any dev-side row leaked past the cutoff.

**No existing script does this without the ingestion step.** This is a new, small entry point
(e.g. `scripts/recompute_monthly_features.py`), not an extension of
`scripts/build_platform_data.py` (which is designed around full re-ingestion and should stay
that way — conflating the two would reintroduce Pitfall 6's confusion between "cheap re-run"
and "network-dependent rebuild").

---

### 5. ADR placement and format — no adr/ directory exists; planner must establish location

**`platform_design/adr/` does not exist.** Confirmed by directory listing this session:
`platform_design/` contains only `full_claude_fable_discussion_20260709_001.txt` and
`platform_design.md`. **`.planning/` has no ADR-specific subdirectory or file naming
convention either** — the closest things are `.planning/UAT-AUDIT-2026-09-09.md` (an audit
log, not an ADR) and `.planning/BASELINE-v1-tracer-bullet.md` (a numbers baseline, not an
ADR). `07-CONTEXT.md` explicitly lists ADR location as "Claude's Discretion."

**The only numbered-ADR convention that DOES exist in this repo is the legacy `CLAUDE.md`'s
embedded "Architecture Decision Records" section (ADR #1 through #12)** — but that convention
is explicitly scoped to the legacy quarterly pipeline (root `CLAUDE.md`, "documents the
separate, frozen legacy quarterly pipeline and its own ADRs — not touched by this phase").
Copying that exact numbering scheme into `platform/` territory would blur a boundary the
project's own two-CLAUDE.md split works hard to keep clear (per RESEARCH.md's "Project
Constraints" section, confirmed this session: "The root CLAUDE.md documents the separate,
frozen legacy quarterly pipeline and its own ADRs ... its conventions ... do not apply here").

**There is also an unrelated `.claude/gsd-core/bin/lib/adr-parser.cjs`** — a GSD tooling
component (part of the GSD command infrastructure itself, not a project-content location) that
parses ADR markdown files with canonical section headers (`status`, `context/goal`,
`decisions`, `considered_options`, etc.) for the `/gsd-adr` family of commands, if any exist.
This is infrastructure, not a place to write phase-specific ADRs, but its header vocabulary
(`## Status`, `## Context`, `## Decision`, `## Considered Options`) is a reasonable, tool-
recognized format to imitate if this project's GSD tooling has an ADR listing/rendering command
that expects it.

**Recommendation for the planner:** create a NEW `platform_design/adr/` directory (parallel to
`platform_design.md`, the authoritative platform design doc this phase's canonical_refs already
cite), with one file per ADR (e.g. `platform_design/adr/0001-l1-feature-policy.md`), using the
GSD-tool-recognized section headers (`## Status`, `## Context`, `## Decision`,
`## Considered Options`, `## Consequences`) so it is at minimum consistent with the parser
infrastructure already in the repo, even though no prior ADR file exists to imitate directly.
This keeps ADRs colocated with the design document they amend, distinct from `.planning/`
(which is phase-tracking/process, not architecture record) and distinct from the legacy
`CLAUDE.md`'s ADR log (which is explicitly the OTHER pipeline's).

---

### 6. Trial registry append — exact call sites, for the ADR's trial-ceiling formula

**`registry.append_trial` signature** (`honesty/registry.py:61-67`, read this session):

```python
def append_trial(
    *,
    config: dict[str, Any],
    features: list[str],
    metrics: dict[str, Any],
    path: Path | str | None = None,
) -> dict[str, Any]:
```

**Two independent call sites, confirmed this session, each firing once per `run_backtest`
invocation:**

1. `driver.py:476-481` — inside `run_backtest` itself, called ONCE at the end of the function
   (after the full walk-forward loop), regardless of `use_regime_tilt`:
   ```python
   registry.append_trial(
       config=trial_config,
       features=list(monthly_features.columns),
       metrics={"n_steps": int(len(equity_curve)), "terminal_log_wealth": terminal_log_wealth},
       path=registry_path,
   )
   ```
2. `no_regime_ablation` (`backtest/baselines.py`) — per RESEARCH.md's traced call chain
   (confirmed against `report.py:573-576`, which calls `no_regime_ablation(...)` as a SEPARATE
   call from the strategy's `run_backtest(...)` at `report.py:569-571`) — `no_regime_ablation`
   delegates to `run_backtest(use_regime_tilt=False)` internally, which hits the SAME
   `driver.py:476-481` call site again, appending a SECOND row.

**Consequence for the ADR's trial-ceiling formula:** every single
`run_full_backtest_evaluation()` call therefore appends **exactly 2 rows** — one from the
`run_backtest(...)` call at `report.py:569-571` (strategy leg), one from the
`no_regime_ablation(...)` call at `report.py:573-576` (ablation leg, itself calling
`run_backtest` a second time). The formula the ADR should state:

```
registry_rows_added = 2 × N_full_evaluation_runs
```

For wave 1: D-03 requires the 13-feature+imputation variant to run once as a logged trial, and
D-02-A's own decision requires re-running the evaluation under the (now 10-column, not 9)
frozen policy — that is a minimum of **2 full-evaluation runs = 4 rows**, not the "~5
trials / ~2 rows" language in `07-CONTEXT.md` D-17, which RESEARCH.md's Pitfall 3 already
flags as an undercount. **The planner should call `registry.read_trials(path=...)`
(`honesty/registry.py:88-96`) immediately before and after wave 1's runs and record the actual
before/after counts in the ADR, rather than trusting any previously-recorded static number**
(30, 34, or any other figure quoted in a planning document) — confirmed this session that the
count has already changed once (30 → 34) between context-gathering and research, purely from
unrelated activity.

---

## Shared Patterns

### Compute-once-thread-through (Pattern 1, RESEARCH.md, confirmed against real code)

**Source:** `evaluation/report.py::run_full_backtest_evaluation`, restructured so
`_reference_label_columns(...)` and `first_decision = dev_features.index[min_train]` are
computed BEFORE `run_backtest(...)` is called, not after (current code computes
`first_decision` from `per_step_metrics["dates"].min()` at `report.py:598`, which is available
only after the loop runs — but is provably equal to `dev_features.index[min_train]` per
`honesty/walkforward.py::expanding_steps`, lines 48-49: `for i in range(min_train, len(index),
step): yield index[i], ...` — the first yielded `i` is always `min_train`).

**Apply to:** `driver.py::run_backtest` (new `frozen_l1_features` kwarg, threaded to every
`_refit_l1` call site) and `report.py::run_full_backtest_evaluation` (compute once, pass to
`run_backtest`, reuse the SAME variable at step (d)'s full-sample fit instead of recomputing).

### Keyword-only optional parameter with `None`-means-old-behavior fallback

**Source:** `driver.py::run_backtest`'s existing `min_train: int | None = None`,
`registry_path: Any = None` (`driver.py:274,277`).

**Apply to:** the new `frozen_l1_features: list[str] | None = None` parameter on both
`run_backtest` and `_refit_l1` — matches house convention exactly, requires no new config
section.

### Skip-real-checkpoint-dependent-test idiom

**Source:** `tests/unit/test_platform_plotting_regime.py:29-32,202-206` —
`REAL_MONTHLY_FEATURES = Path(...)` + `@pytest.mark.skipif(not REAL_MONTHLY_FEATURES.exists(), ...)`.

**Apply to:** the criterion-1 equivalence test's real-checkpoint-dependent variant, and any new
test in `test_platform_backtest_driver.py` that needs the real `monthly_features` checkpoint.

### Registry-count-at-execution-time, never a static number

**Source:** RESEARCH.md's own "Don't Hand-Roll" table entry for the deflated-Sharpe trial
count — `registry.read_trials(path=...)` re-read live, never trusted from a planning document.

**Apply to:** the ADR's trial-ceiling section (finding #6 above).

---

## No Analog Found

| File/Artifact | Role | Data Flow | Reason |
|---|---|---|---|
| `platform_design/adr/*.md` (new ADR) | documentation | n/a | No ADR file or directory precedent exists anywhere in this repo outside the legacy `CLAUDE.md`'s embedded, explicitly-out-of-scope numbered log. Planner must establish format (recommendation: finding #5 above). |
| Golden-constant re-pin comment convention | test (regression) | n/a | No precedent found for re-pinning an `EXPECTED_*` constant with a dated, decision-naming comment. Planner must establish (recommendation: finding #3 above). |
| `scripts/recompute_monthly_features.py` (new) | script | batch, offline | `scripts/build_platform_data.py` is the only sibling script and does the OPPOSITE (full network rebuild) — a new, narrowly-scoped entry point is needed (finding #4 above), not an extension. |

---

## Wave 2 — not this pass (minimal appendix, per orchestrator instruction)

Flagged only because it surfaced incidentally while reading files in scope:

- `labeling/jump_model.py::canonicalize_states`'s default sort key (`trailing_return_1m`) is one
  of classifier #1's 13 raw columns, which D-10 (wave 2) excludes from classifier #2's feature
  set by construction — every classifier #2 fit will hit the function's "centroid column 0"
  fallback path (RESEARCH.md Pitfall 4, independently confirmed by reading
  `jump_model.py:214-221` this session: the fallback fires with a WARNING log when
  `trailing_return_1m` is absent from the fit's feature set). Not actionable in wave 1; noted
  here only so wave 2's planning pass does not need to rediscover it from scratch. No pattern
  work done for it in this pass.
- `src/trading_crab_lib/momentum.py` / `divergence.py` — confirmed (RESEARCH.md, this session)
  to be **pattern source only, must be ported not imported** into `platform/` for wave 2's
  relative-strength features (criterion 8's constraint). Not touched by this pass.
