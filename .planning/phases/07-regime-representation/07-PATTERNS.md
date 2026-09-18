# Phase 7 (Wave 2): Regime Representation - Pattern Map

**Mapped:** 2026-09-15
**Scope:** WAVE 2 ONLY — criteria 5, 6, 7 + INV-01. Criteria 1–4 (wave 1) are closed; this
file **overwrites** wave 1's pattern map. Wave 1's patterns are preserved in its SUMMARYs and
`platform_design/adr/0001-l1-feature-policy.md` — not repeated here.

**Files analyzed:** 7 new/modified files (+ config additions)
**Analogs found:** 5 / 7 direct analogs; 1 partial (no analog — new convention needed); 1 "no
code exists yet" (deflated Sharpe)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `platform/labeling/jump_model.py` (edit: `canonicalize_states`) | utility (labeling) | transform | itself, pre-edit (4 existing call sites) | exact — additive signature change |
| `platform/features/relative.py` (new) | utility (feature engineering) | transform / batch | `src/trading_crab_lib/momentum.py` (legacy, **pattern-only, do not import**) | role-match, cadence differs (quarterly→monthly) |
| `platform/evaluation/dependence.py` (new, or extend `disagreement.py`) | service (evaluation) | transform | `platform/evaluation/disagreement.py::measure_label_disagreement` | exact |
| `platform/evaluation/deflated_sharpe.py` (new) | utility (statistics) | transform | none in-repo — Bailey–López de Prado (2014) formula, hand-implemented | **no analog found** |
| `platform/allocation/joint_tilt.py` (new) | service (allocation) | transform | `platform/allocation/tilt.py::regime_tilt_weights` / `vol_targeted_tilt` (composed, not extended) | partial — composition of two existing pure functions, blend step itself is new |
| `platform/honesty/registry.py` (edit or new helper `total_trial_count`) | utility (honesty) | CRUD (read) | `read_trials()` (same file, extend) | exact |
| `config/platform_settings.yaml` (edit: `fred_monthly.series` + M2SL/TOTALSL) | config | batch (ingestion) | `config/platform_settings.yaml:21-48` (existing `fred_monthly.series` block, e.g. `WTISPLC`) | exact |
| `tests/unit/test_platform_labeling.py` (extend) | test | request-response | existing `TestCanonicalizeStates` class in same file | exact |
| `tests/unit/test_platform_features_relative.py` (new) | test | transform | `tests/unit/test_platform_ingestion_macro_monthly.py` (fixture/mocking shape) | role-match |
| `tests/unit/test_platform_evaluation_dependence.py` (new) | test | transform | `tests/unit/test_platform_evaluation_disagreement.py` (if exists) or `test_platform_labeling.py`'s oracle-test shape | role-match |
| `tests/unit/test_platform_evaluation_deflated_sharpe.py` (new) | test | transform | Phase 3's DP-decode oracle test (brute-force-vs-formula pattern, cited in RESEARCH) | pattern-only |
| `tests/unit/test_platform_allocation_joint_tilt.py` (new) | test | transform | `tests/unit/test_platform_allocation_tilt.py` (if exists — same module family) | role-match |

---

## Pattern Assignments

### 1. `canonicalize_states` — add `sort_column` keyword-only parameter

**Analog:** the function itself, pre-edit — `src/trading_crab_lib/platform/labeling/jump_model.py:214-221` (verified this session per RESEARCH Code Examples).

**Current (the landmine):**
```python
# jump_model.py:214-221
if "trailing_return_1m" in feature_names:
    sort_col = feature_names.index("trailing_return_1m")
else:
    sort_col = 0
    log.warning(
        "trailing_return_1m not in feature_names — falling back to centroid "
        "column 0 for canonicalization sort order"
    )
```

**Target shape (RESEARCH Pattern 1, quoted verbatim as the recommended fix):**
```python
def canonicalize_states(
    states: np.ndarray,
    centroids: np.ndarray,
    feature_names: list[str],
    *,
    sort_column: str = "trailing_return_1m",
) -> tuple[np.ndarray, np.ndarray]:
    if sort_column not in feature_names:
        raise ValueError(
            f"sort_column={sort_column!r} not in feature_names — a canonicalization "
            "sort column must be present in the fitted feature set. Pass the caller's "
            "own defining column explicitly (never rely on a silent fallback)."
        )
    sort_col = feature_names.index(sort_column)
    order = np.argsort(centroids[:, sort_col])
    remap = {old: new for new, old in enumerate(order)}
    new_states = np.array([remap[s] for s in states])
    return new_states, centroids[order]
```

**Transfers cleanly — confirmed:**
- Keyword-only, defaulted parameter matches this codebase's established shape for additive,
  backward-compatible signature changes: `backtest/driver.py`'s `frozen_l1_features` and
  `trial_tag` parameters, and `run_backtest`'s `min_train: int | None = None` (both cited by
  the orchestrator as the precedent to match). This edit follows that shape exactly —
  keyword-only via `*`, defaulted to reproduce current behavior byte-for-byte.
- All 4 existing call sites (`driver.py::_refit_l1` line 262, `evaluation/report.py` line 924,
  `labeling/diagnostics.py::label_regimes` line 286, and one more — RESEARCH says "4 existing
  call sites" but names 3; verify the 4th at implementation time) never pass `sort_column`
  today, so the default reproduces current behavior. Classifier #1 always has
  `trailing_return_1m` in its frozen 10-column set (D-02-A), so it can never hit the new
  `ValueError`.
- **No existing test pins the old warning-and-fallback behavior** — confirmed by RESEARCH via
  grep across `tests/unit/test_platform_labeling.py`: the file's only `caplog`-based WARNING
  test (`TestReportDiagnosticsReportOnly::test_violation_warns_but_does_not_raise`) covers a
  *different* warning (the §4.4 occupancy floor), not this fallback. **No re-pin is needed**
  for this specific change (contrast with the change-point re-pin wave 1 required — that one
  DID have a pinned regression test; this one does not).

**Test analog:** extend the existing `TestCanonicalizeStates` class in
`tests/unit/test_platform_labeling.py` with two new cases (RESEARCH Code Examples, quoted):
one asserting no fallback warning fires when `sort_column` is present, one asserting
`pytest.raises(ValueError, match="sort_column")` when absent.

---

### 2. Window-constant re-derivation for monthly cadence (relative-strength port)

**Analog (platform-native, for module shape/conventions):**
`src/trading_crab_lib/platform/ingestion/macro_monthly.py:1-30` (docstring) — this is the
established in-repo precedent for "the legacy version of this exists at quarterly cadence;
this is its monthly analog, ported not imported, with the resample rule and every numeric
constant re-derived, not copied":

```python
# platform/ingestion/macro_monthly.py:1-13 (docstring, verbatim)
"""
Monthly macro/long-history raw ingestion (DATA-01).

The incumbent quarterly pipeline's fetchers (``ingestion/fred.py``,
``ingestion/multpl.py``, ``ingestion/macrotrends.py``) all hardcode a
period-end quarterly resample rule internally — reusing them verbatim would
silently keep quarterly cadence and defeat this phase's entire purpose
(RESEARCH Pitfall 1).
This module writes thin monthly analogs that reuse the same client
construction / parallel-fetch / scrape-and-parse patterns but target
``"ME"`` (month-end) instead, without editing any frozen incumbent file
(D-01).
"""
```

This is **exactly** the shape `platform/features/relative.py` must follow: same
client-construction/algorithm pattern as the legacy source, resample/window unit re-derived
for monthly, ported not imported, header comment naming the source module and the reason.

**Legacy source (pattern only — port the algorithm, never import):**
`src/trading_crab_lib/momentum.py` — verified this session to have **zero**
`trading_crab_lib`-internal imports (only `numpy`, `pandas`, `logging`, `typing`), so the
port itself cannot widen the legacy-import ratchet as long as the import statement (not the
function body) is not copied.

| Function | Legacy default (quarterly) | Re-derived default (monthly) | Source lines |
|---|---|---|---|
| `compute_relative_strength` pairs | n/a (ratio math, unit-agnostic) | unchanged — pure ratio | `momentum.py:77-110` |
| `compute_rolling_cross_correlation` window | `window=8` "quarters" (≈24 months) | **24** months | `momentum.py:123-157` |
| `compute_trailing_momentum` windows | `windows=[2, 4, 8]` (quarters ≈ 6/12/24mo) | **[6, 12, 24]** months | (same module, per RESEARCH Pitfall 5) |
| `compute_inflation_acceleration` | period-agnostic (2nd derivative) | unchanged — no window constant | `momentum.py:162-180` |

**Concrete port instruction:** copy function **bodies** (not `from momentum import ...`) into
`platform/features/relative.py`, with an attribution comment naming source file + line range
(mirrors the pattern used elsewhere in this repo for "ported, never imported" seams — contrast
`platform_settings.yaml`'s macrotrends comment, `config/platform_settings.yaml:103-104`,
`"Reused verbatim from trading_crab_lib.ingestion.macrotrends (D-01: import, never edit)"` —
that is the OPPOSITE convention, for a seam that IS imported; do not confuse the two).
Rename every docstring "quarterly" reference to "monthly." **Do not port** `compute_rrg` /
`rolling_zscore` / `percentile_rank` / `normalize_100` from legacy `diagnostics.py` — those are
tactical/asset-rotation concepts, not regime-labeling inputs, and no D-10/D-11/D-12 candidate
references them.

**Ratchet guard (must not regress):** `tests/unit/test_platform_legacy_import_ratchet.py`
performs a whole-tree AST scan (not a diff of new files) pinned at **31**. Adding even one
`from trading_crab_lib.momentum import ...` anywhere under `platform/` — including inside the
new module — fails it immediately. Verify with
`pytest tests/unit/test_platform_legacy_import_ratchet.py -v` after the port lands.

---

### 3. `blend_regime_tilts()` — genuinely new; closest analog and contract to preserve

**Confirmed (per orchestrator + RESEARCH, both independently verified this session):**
`platform/allocation/tilt.py::vol_targeted_tilt` and `regime_tilt_weights` each accept
exactly **one** `regime_or_probs` argument — no analog exists for "two probability inputs,
blended." **This is new code; say so plainly, per the task's own instruction.**

**Closest analog (for call shape/composition, not for the blend logic itself):**
`platform/allocation/tilt.py` (full file read by RESEARCH this session) — the composable
three-function pipeline `regime_tilt_weights(...)` → `portfolio_vol(...)` →
`vol_target_scale(...)`. The new `blend_regime_tilts` must **reuse `portfolio_vol` and
`vol_target_scale` unmodified**, inserting only a new pre-scaling weight-blend step:

```python
# NEW — platform/allocation/joint_tilt.py (RESEARCH Pattern 4, quoted)
def blend_regime_tilts(
    probs_1: pd.Series, returns_by_regime_1: pd.DataFrame,
    probs_2: pd.Series, returns_by_regime_2: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    weight_1: float = 0.5,     # fixed, pre-declared — NOT swept (D-13's spirit extended)
    target_vol_annual: float, halflife: float, min_obs: int,
) -> dict:
    """D-14: two SEPARATE probability-weighted tilts, blended at the WEIGHT
    level, never a product (state_1, state_2) cell."""
    from trading_crab_lib.platform.allocation.tilt import (
        regime_tilt_weights, portfolio_vol, vol_target_scale,
    )
    tilt_1 = regime_tilt_weights(probs_1.idxmax(), returns_by_regime_1, probs_1)
    tilt_2 = regime_tilt_weights(probs_2.idxmax(), returns_by_regime_2, probs_2)
    blended = (weight_1 * tilt_1).add((1 - weight_1) * tilt_2, fill_value=0.0)
    total = blended.sum()
    base_weights = blended if total <= 0 else blended / total
    if base_weights.empty or base_weights.sum() <= 0:
        return {"weights": pd.Series(dtype=float), "cash": 1.0, "scale": 0.0}
    port_vol = portfolio_vol(base_weights, asset_returns, halflife=halflife, min_obs=min_obs)
    scale = vol_target_scale(target_vol_annual, port_vol)
    return {"weights": base_weights * scale, "cash": 1.0 - scale, "scale": scale}
```

**Output contract to preserve (from `vol_targeted_tilt`, verified this session):** weights
sum to `scale` (`scale ≤ 1`, `cash = 1 - scale`), long-only clipping upstream in
`regime_tilt_weights` — the blend function's return dict shape (`{"weights", "cash",
"scale"}`) mirrors `vol_targeted_tilt`'s own return shape so downstream consumers (backtest
driver, reporting) need no special-casing for "joint" vs "single" tilts.

**Blend weight must be fixed and declared in the ADR before either classifier's
walk-forward runs** (0.5 here is illustrative, not prescriptive) — sweeping it is an
unregistered selection dimension the trial ceiling (D-17/Pitfall 8) does not budget for.

**Wiring choice for the planner (RESEARCH's explicit recommendation):** two independent
`run_backtest()` calls (one per classifier), combined post-hoc via `blend_regime_tilts` at
each shared decision date — **not** an invasive extension of `driver.py`'s per-step loop.
Reasoning: `run_backtest` is heavily tested load-bearing code; composition over two calls to
already-tested code keeps per-classifier attribution trivial (mirrors D-04's own reasoning
for wave 1's frozen-cols threading).

**Test analog:** no `test_platform_allocation_joint_tilt.py` file exists yet (Wave 0 gap per
VALIDATION.md) — model it on whatever existing `tests/unit/test_platform_allocation_*.py`
covers `vol_targeted_tilt`/`regime_tilt_weights` today (same fixture shape: synthetic
`returns_by_regime` + `asset_returns` DataFrames), adding degenerate-input cases (empty
probs, single classifier weight_1=1.0 reduces to `vol_targeted_tilt` exactly).

---

### 4. `measure_labeling_dependence()` — dependence module

**Analog (platform, imitate directly):** `platform/evaluation/disagreement.py` — the closest
sibling, built in wave 1. It returns a crosstab and carries `suspicious`/`suspicious_reason`
fields. Confirmed shape via `label_disagreement` (in `platform/plotting/regime.py`, called by
`disagreement.py`): aligns two label Series on common index, returns `per_state_confusion`
(a `pd.crosstab`) plus `n_compared`, and defensively coerces `state_N`-string columns while
guarding the silent-zero-`n_compared` trap.

**Template to extend (RESEARCH Pattern 3, quoted — reuses the crosstab, does not
reimplement alignment):**
```python
# NEW — platform/evaluation/dependence.py (or extend disagreement.py)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats.contingency import association

def measure_labeling_dependence(states_1: pd.Series, states_2: pd.Series) -> dict:
    """D-15: several statistics, no pre-declared threshold. Reuses
    label_disagreement's alignment/crosstab exactly — never a second
    alignment implementation."""
    from trading_crab_lib.platform.plotting.regime import label_disagreement
    base = label_disagreement(states_1, states_2)   # reuses the SAME crosstab machinery
    if base["n_compared"] == 0:
        return {**base, "adjusted_rand": float("nan"), "nmi": float("nan"), "cramers_v": float("nan")}
    common = states_1.index.intersection(states_2.index)
    a = states_1.loc[common].to_numpy()
    b = states_2.loc[common].to_numpy()
    return {
        **base,
        "adjusted_rand": adjusted_rand_score(a, b),
        "nmi": normalized_mutual_info_score(a, b),
        "cramers_v": float(association(pd.crosstab(a, b).to_numpy(), method="cramer")),
    }
```

All three statistics (`adjusted_rand_score`, `normalized_mutual_info_score`,
`association(..., method="cramer")`) were live-verified this session against a synthetic
6-element example (ARI=0.1667, NMI=0.5794, Cramér's V=0.5 for a 2×2 table) — all run
correctly in this environment; no new dependency required (`sklearn` 1.9.1, `scipy` 1.17.1
already installed and already used elsewhere in `platform/`).

**No pre-declared threshold (D-15) — report all three side by side.** ARI (pairwise
clustering-agreement, chance-corrected), NMI (information-theoretic, robust to different K),
Cramér's V (classical-statistics effect size, single unit-interval number for "at a glance").
A directional statistic (Theil's U) is explicitly **not** recommended as a fourth addition.

**"Suspiciously clean" check (from VALIDATION.md, mirrors `disagreement.py`'s own
`suspicious`/`suspicious_reason` posture):** ARI or NMI exactly 1.0 → same underlying labels,
a wiring bug. All three exactly 0.0 simultaneously → alignment-bug suspicion. Neither is a
pass/fail gate; both are a "confirm this isn't a bug" prompt.

---

### 5. Deflated Sharpe — no analog exists; new module, sketch provided

**Confirmed: nothing exists.** Grep-verified this session: five `deflat*` hits in `src/`, all
comments/docstrings, zero function definitions.

**Location recommendation:** `platform/evaluation/deflated_sharpe.py`, alongside
`platform/evaluation/kpis.py` (sibling evaluation-statistics module) — follows the existing
one-concept-per-file convention already used by `disagreement.py` and (new) `dependence.py`.

**House pattern for a pure statistical helper with a hand-worked-oracle test:** the strongest
verification precedent in this project is Phase 3's DP-decode oracle test (brute-force
enumeration proven identical to the formula across 7 cases) — reach for the same shape here:
`test_platform_evaluation_deflated_sharpe.py` should assert the DSR formula against a
hand-computed small-N example (N=1 trial → reduces toward raw significance; N→∞ → DSR→toward
the null), not merely a shape/existence check (VALIDATION.md's "Evidence-Shape Requirement"
explicitly names this project's two prior burns from existence/shape-only checks).

**Sketch (RESEARCH Code Examples, quoted — a starting point, not a final implementation; the
exact `sharpe_variance` estimator is an open ADR decision, not pinned by this sketch):**
```python
# NEW — platform/evaluation/deflated_sharpe.py
# formula per Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"
import numpy as np
from scipy.stats import norm

def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    euler_mascheroni = 0.5772156649
    if n_trials <= 1:
        return 0.0
    return np.sqrt(sharpe_variance) * (
        (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

def deflated_sharpe_ratio(
    observed_sharpe: float, n_trials: int, sharpe_variance: float,
    skew: float, kurtosis: float, n_obs: int,
) -> float:
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    denom = np.sqrt(1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2)
    z = (observed_sharpe - sr0) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))
```
Style must still follow house conventions not shown in the sketch: `from __future__ import
annotations`, `log = logging.getLogger(__name__)`, type hints on all public functions,
`# ── Section ──` dividers if the file grows multiple logical blocks, docstring citing the
paper (Bailey & López de Prado 2014, SSRN 2460551).

**`total_trial_count()` — the provenance-header reader it must call:**

Template/analog: `platform/honesty/registry.py::read_trials` — quoted in full below (its
"actual shape," per the task instruction) — a bare, header-unaware reader that the new
function must wrap, not modify:

```python
# registry.py — read_trials is a bare pd.read_json(..., lines=True), no header-awareness
# (verified this session by reading registry.py in full)
```

**The live ledger's header row's actual shape** (verified this session, `wc -l
registry/trials.jsonl` = **1**):
```json
{"config_hash": "RESET", "config": {"trial_tag": "REGISTRY-RESET-P7W1", "record_type": "provenance_header", "reset_reason": "...", "archived_to": "registry/archive/trials-pre-P7W1-reset.jsonl", "archived_row_count": 42, "prior_genuine_trials": 38, "discarded_smoke_rows": 4, "deflation_note": "..."}, "features": [], "metrics": {"prior_genuine_trials": 38, "discarded_smoke_rows": 4}, "git_sha": null, "timestamp": "2026-09-15T14:43:46.377815+00:00"}
```
`config.record_type == "provenance_header"` is the load-bearing discriminator.
`config.prior_genuine_trials == 38` is the count to add to post-header rows.

**Correct reader (RESEARCH Pitfall 4, quoted — write as its own tested function, do not
inline into the DSR module):**
```python
def total_trial_count(path=None) -> int:
    df = registry.read_trials(path)
    if df.empty:
        return 0
    is_header = df["config"].apply(lambda c: isinstance(c, dict) and c.get("record_type") == "provenance_header")
    prior = int(df.loc[is_header, "config"].apply(lambda c: c["prior_genuine_trials"]).sum()) if is_header.any() else 0
    return prior + int((~is_header).sum())
```
**Warning sign to avoid:** `len(registry.read_trials())` directly, or hardcoding `38`/`42` as
a literal offset — both repeat the "34, then 38, then 42 — all stale the moment you write them
down" failure wave 1's own ADR documents.

---

### 6. INV-01 ingestion — M2SL and TOTALSL

**Analog (platform, imitate directly):** `config/platform_settings.yaml:21-48`, the existing
`fred_monthly.series` block — same shape used for e.g. `WTISPLC`:

```yaml
# config/platform_settings.yaml:21-48 (existing block, pattern to copy)
fred_monthly:
  series:
    GS10:
      name:  "fred_gs10"
      shift: false
    ...
    WTISPLC:
      name:  "wti_fred"        # oil cross-check vs macrotrends wti_crude
      shift: false
```

**New entries to add (same block, same shape):**
```yaml
    M2SL:
      name:  "fred_m2sl"
      shift: false
    TOTALSL:
      name:  "fred_totalsl"
      shift: false
```

**Fetch code path (analog, platform-native, imitate directly):**
`platform/ingestion/macro_monthly.py::fetch_fred_monthly` / `_fetch_fred_monthly` — already
generic over `cfg["fred_monthly"]["series"]`; adding the two new keys to the YAML is
sufficient, **no new Python code required** for the fetch itself (verified: the function
iterates the config dict, no per-series special-casing beyond `shift`).

**Alignment/interpolation — none needed, contrary to `BCNSDODNS`:** live-verified this
session via `fredapi.Fred.get_series()`:

| Series | First valid | n (non-null) | Native frequency |
|---|---|---|---|
| `M2SL` | 1959-01-01 | 811 | Monthly — matches spine natively |
| `TOTALSL` | 1943-01-01 | 1003 | Monthly — matches spine natively |
| `BCNSDODNS` (considered, rejected) | 1945-10-01 | 305 | **Quarterly** (would need `fred_gdp`-style forward-fill) |
| `TOTBKCR` (considered, rejected) | 1973-01-03 | 2801 | Starts too late (1973, not 1962) |

Both `M2SL` and `TOTALSL` are natively monthly and require **no** interpolation or
alignment treatment — no `fred_gdp`-style quarterly-repeat-across-months handling is needed,
unlike the rejected `BCNSDODNS` candidate (which would need the same forward-fill
`monthly_raw["fred_gdp"]` already gets, e.g. 3758.147 repeated across 1962-02/03/04).
**Document `BCNSDODNS` as considered-and-rejected in the ADR** (frequency mismatch),
consistent with wave 1 ADR's D-03 imputation-trial treatment (state what a rejected option
would have cost, don't silently drop it).

**Test analog:** extend `tests/unit/test_platform_ingestion_macro_monthly.py` (existing file,
per VALIDATION.md's requirements map) with new cases for `M2SL`/`TOTALSL`, mirroring whatever
per-series mocked-fetch test pattern already covers `GS10`/`WTISPLC` in that file, plus one
live smoke fetch (VALIDATION.md: "new cases ... + one live fetch").

---

## Shared Patterns

### Legacy-import ratchet (applies to every new `platform/` file)
**Source:** `tests/unit/test_platform_legacy_import_ratchet.py` — whole-tree AST scan pinned
at **31**, may only decrease. **Apply to:** every new file under `platform/`, especially
`features/relative.py` (the porting temptation). Never `from trading_crab_lib.momentum import
...` or `.diagnostics import ...` anywhere under `platform/` — copy function bodies with
attribution comments instead.

### Report-only, never-gate posture (D-02/D-07/D-15)
**Source:** `platform/labeling/diagnostics.py::occupancy_and_sojourns` /
`report_labeling_diagnostics` (unmodified, reused for classifier #2's own occupancy). **Apply
to:** `measure_labeling_dependence` (D-15: no pass/fail gate) and the deflated-Sharpe report
(D-07 lineage: bands are plausibility checks, never a quality gate).

### `from __future__ import annotations`, type hints, `log = logging.getLogger(__name__)`,
no bare `except:`, `pathlib.Path`, ruff 127-col, `# ── Section ──` dividers
**Source:** house convention, visible in every file read this session
(`macro_monthly.py`, `tilt.py`, `jump_model.py`, `registry.py`). **Apply to:** all 5 new
files (`relative.py`, `dependence.py`, `deflated_sharpe.py`, `joint_tilt.py`, any
`registry.py` extension).

### Config additions read defensively via `cfg.get()`, never added to required-sections list
**Source:** Phase 2/4 pattern, cited by RESEARCH's Established Patterns. **Apply to:** the new
`fred_monthly.series` entries (M2SL/TOTALSL) and any new `labeling_2`/`allocation.blend`
config section for classifier #2's K/λ and the blend weight.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `platform/allocation/joint_tilt.py::blend_regime_tilts` | service (allocation) | transform | Confirmed by both orchestrator and RESEARCH: nothing in `allocation/tilt.py` accepts two probability inputs today. The planner establishes this convention; `vol_targeted_tilt`'s output contract (weights sum to scale, long-only) is the constraint to preserve, not a template to extend. |
| `platform/evaluation/deflated_sharpe.py` | utility (statistics) | transform | Grep-confirmed: zero function definitions anywhere in `src/` (5 hits, all comments/docstrings). New code from a paper formula, not an in-repo pattern. |

---

## Metadata

**Analog search scope:** `src/trading_crab_lib/platform/{labeling,evaluation,allocation,
honesty,ingestion}/`, `config/platform_settings.yaml`, `src/trading_crab_lib/momentum.py`
(legacy, pattern-source only), `tests/unit/test_platform_*.py`, `platform_design/adr/`.
**Files scanned (read in full or targeted this session, per RESEARCH's own verification
log):** `jump_model.py`, `disagreement.py`, `tilt.py`, `registry.py`, `macro_monthly.py`,
`platform_settings.yaml`, `momentum.py`, `diagnostics.py` (legacy), `driver.py` (relevant
sections), `test_platform_legacy_import_ratchet.py`.
**Pattern extraction date:** 2026-09-15.
**Everything in this file is grounded with `path:line` citations reproduced from
`07-RESEARCH.md`'s own live-verified session claims; no analog is asserted without either a
quoted excerpt or an explicit "no analog found."**
