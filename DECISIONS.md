# Trading-Crab — Development Decisions Log

A chronological log of judgment calls that don't rise to the level of a formal
Architecture Decision Record but are important for future contributors (human or AI)
to know about. Each entry records what was decided, why, and what was explicitly
rejected.

For major architectural decisions (fixed PCA components, two feature files, balanced
KMeans, etc.) see `ARCHITECTURE.md`. For gotchas and anti-patterns see `PITFALLS.md`.

---

## 2026-03-16

### D1. GSD `pipelines_from_gsd_version/05_predict.py` — NOT adopted

**Decision:** the GSD-generated `05_predict.py` (bundle API) was reviewed and
explicitly rejected. The existing `pipelines/05_predict.py` (flat API) is canonical.

**Why rejected:** the GSD version called `train_current_regime(X, y, cv_splits=N)`
from `prediction/classifier.py`, which returns a bundle dict
`{"models": {"rf": ..., "dt": ...}, "cv_reports": {...}}` and saves that dict as
`current_regime.pkl`. Two downstream consumers cannot handle a bundle dict:
- `pipelines/07_dashboard.py` does `hasattr(current_model, "feature_names_in_")`
  assuming a bare sklearn estimator.
- `run_pipeline.py step5_predict` calls `predict_current(current_model, X)` which
  requires a sklearn estimator, not a dict.

Adopting the GSD version would have required simultaneous changes to both consumers,
with no immediate benefit. The flat API is simpler and sufficient.

**What WAS adopted from GSD version:**
- The monkeypatch fix in `pipelines/02_features.py`: changed
  `from market_regime.transforms import engineer_all` (direct import, unpatchable)
  to `from market_regime import transforms as _transforms_module` (module-level
  reference, patchable by `monkeypatch.setattr`). This makes the step-02 smoke test
  work correctly.

---

### D2. `prediction/` converted from flat module to package

**Decision:** `src/market_regime/prediction.py` (single file) was converted to a
package: `src/market_regime/prediction/__init__.py` + `prediction/classifier.py`.

**Why:** new test files (`test_models_regime.py`, `test_models_reporting.py`) imported
from `market_regime.prediction.classifier`. Python cannot treat a flat `.py` file as
a package with sub-modules — `'market_regime.prediction' is not a package`.

**Split:** the existing flat-API content moved intact to `__init__.py`. A new
`classifier.py` was created with the backwards-compatible bundle API (FoldReport
namedtuple, bundle-returning `train_current_regime`, `train_forward_classifiers`,
and a tri-modal `model_metrics_summary`).

**Invariant:** `__init__.py` is the production API; `classifier.py` is the test
layer. See ARCHITECTURE.md ADR #12.

---

### D3. `make_behavior_labels` changed to strict inequalities

**Decision:** changed `r >= up_threshold` / `r <= down_threshold` to
`r > up_threshold` / `r < down_threshold` in `prediction/__init__.py`.

**Why:** with both thresholds at `0.0` (the default in `train_forward_behavior_models`),
a return of exactly `0.0` was incorrectly classified as `"up"` (because `0.0 >= 0.0`).
The intended semantics are: "strictly exceeds the threshold to be directional;
at-the-threshold is ambiguous / flat."

**Impact:** extremely rare in practice on real price data (return == exactly 0.0 to
floating-point precision is vanishingly unlikely). Only affects synthetic test data
and degenerate edge cases.

---

### D4. GSD `pipelines_from_gsd_version/01_ingest.py` and `02_features.py` — NOT adopted (deferred)

**Decision:** the GSD wrappers for steps 01 and 02 (which add `--refresh`, `--verbose`,
`--market-code` CLI flags and delegate to `run_pipeline.step1_ingest()`) were reviewed
but not applied to `pipelines/01_ingest.py` and `pipelines/02_features.py`.

**Why deferred:** the improvement is real (CLI flags + single source of truth for step
logic), but not urgent — `run_pipeline.py --steps 1,2` already provides the same
functionality with all flags. The standalone scripts are used mainly for quick manual
runs. This is a low-priority cosmetic improvement, not a correctness fix.

**When to revisit:** if `run_pipeline.py step1_ingest()` and `step2_features()` are
significantly changed, update the standalone scripts to delegate to them at the same time.

---

### D5. Pipeline smoke tests use `tmp_path` — checkpoint contamination eliminated

**Decision:** `tests/test_pipelines_ingest_features.py` redirects all file I/O to
pytest's `tmp_path` fixture using `monkeypatch.setattr(step, "DATA_DIR", tmp_path)`.

**Why:** previously the smoke tests called `step01.main([])` and `step02.main()`
with `DATA_DIR` pointing at the real `data/` tree. This overwrote
`data/checkpoints/macro_raw.parquet` (and `features_causal`, `features_noncausal`)
with synthetic 4-row test data, requiring `python run_pipeline.py --recompute` after
every `pytest` run and causing tracked checkpoint files to appear modified.

**How it works:** `step01.DATA_DIR` and `step02.DATA_DIR` are module-level attributes.
`monkeypatch.setattr` patches them per-test in the loaded module's namespace, so the
script writes to `tmp_path/raw/`, `tmp_path/processed/` etc. without touching production
data. The patch is automatically reverted by monkeypatch after each test.

**Result:** no production checkpoint files are written during `pytest`. The committed
synthetic checkpoint files in `data/checkpoints/` were removed (no longer needed).

---

### D6. `pipelines_from_gsd_version/` kept in repo (per owner decision)

**Decision:** the `pipelines_from_gsd_version/` directory was kept in the repository
rather than deleted, per the repository owner's preference.

**Note for future sessions:** these scripts represent an alternative pipeline design
explored via the GSD framework. The decisions about which changes to apply are
documented in D1 and D4 above. Do not treat these as "more current" than `pipelines/`.

---

### D7. `legacy/` kept in repo (per owner decision)

**Decision:** the `legacy/` directory (including modular scripts and notebooks) was kept
in the repository rather than trimming it to `unified_script.py` only.

**Note for future sessions:** `legacy/unified_script.py` is the algorithm ground truth.
The modular `legacy/*.py` files are an intermediate refactoring step between the monolith
and `src/` — they are not independently authoritative. When implementing a remaining gap
(e.g., empirical forward probabilities), refer to `unified_script.py`, not the modular
files, to avoid inconsistencies.
